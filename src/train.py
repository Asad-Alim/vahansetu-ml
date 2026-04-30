

"""
train.py — Batch PPO training optimised for CPU.
"""

import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from visualizer import plot_training_log, plot_batch_summary, print_training_summary

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from data_loader  import build_dataset
from alns_env     import ALNSEnv
from constraints  import check_route
from visualizer   import plot_training_log, plot_batch_summary


class LoggingEnv(ALNSEnv):
    def __init__(self, instances, max_iter, seed, episode_log):
        super().__init__(instances, max_iter=max_iter, seed=seed)
        self.episode_log = episode_log
        self._ep_reward  = 0.0
        self._ep_num     = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ep_reward = 0.0
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        self._ep_reward += reward
        if done:
            result = check_route(self.best.route, self.orders_map, self.instance.vehicle)
            self.episode_log.append({
                "episode":    self._ep_num,
                "reward":     round(self._ep_reward, 4),
                "best_cost":  round(self.best.cost(), 2),
                "init_cost":  round(self.init_cost, 2),
                "violations": result.deadline_violations,
            })
            self._ep_num += 1
            self._ep_reward = 0.0
        return obs, reward, done, trunc, info


def train_batch(args):
    instances_dir = os.path.join(args.data_dir, "instances")
    print(f"\nLoading instances from: {instances_dir}")
    instances = build_dataset(instances_dir, max_files=args.max_files)
    if not instances:
        print("ERROR: No instances found."); return

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    steps_per_batch = 10240
    print(f"Each batch = {steps_per_batch} steps | max_iter={args.max_iter} | instances={len(instances)}")

    episode_log     = []
    batch_summaries = []
    checkpoint_path = os.path.join(args.model_dir, "ppo_alns_checkpoint")
    batch_start     = 1

    def make_env_fn(seed):
        def _init():
            return LoggingEnv(instances, args.max_iter, seed=seed, episode_log=episode_log)
        return _init

    vec_env = make_vec_env(make_env_fn(42), n_envs=1)

    if args.resume and os.path.exists(checkpoint_path + ".zip"):
        print(f"Resuming from: {checkpoint_path}.zip")
        model = PPO.load(checkpoint_path, env=vec_env)
        sp = os.path.join(args.log_dir, "batch_summaries.json")
        if os.path.exists(sp):
            with open(sp) as f:
                batch_summaries = json.load(f)
            batch_start = len(batch_summaries) + 1
        print(f"Resuming from batch {batch_start}")
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            device="cpu",
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            learning_rate=3e-4,
            # n_steps=128,
            # batch_size=16,
            n_steps=512,        # was 128 — now fits ~10 complete episodes per rollout
            batch_size=64,      # was 16 — scale up with n_steps
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

    total = args.batches
    print(f"Starting {total} batch(es)")
    print(f"Estimated time per batch: 15-25 min on CPU\n")

    for batch_num in range(batch_start, batch_start + total):
        print(f"\n{'='*60}")
        print(f"  BATCH {batch_num} / {batch_start + total - 1}")
        print(f"{'='*60}")

        before  = len(episode_log)
        t_start = time.time()

        model.learn(
            total_timesteps=steps_per_batch,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        elapsed = (time.time() - t_start) / 60
        new_eps = episode_log[before:]

        if new_eps:
            avg_reward = np.mean([e["reward"]     for e in new_eps])
            avg_viols  = np.mean([e["violations"]  for e in new_eps])
            avg_improv = np.mean([
                (e["init_cost"] - e["best_cost"]) / max(e["init_cost"], 1) * 100
                for e in new_eps
            ])
        else:
            avg_reward = avg_viols = avg_improv = 0.0

        print(f"\n  Batch {batch_num} done in {elapsed:.1f} min")
        print(f"  Episodes : {len(new_eps)}")
        print(f"  Avg reward      : {avg_reward:.4f}")
        print(f"  Avg improvement : {avg_improv:.1f}%")
        print(f"  Avg violations  : {avg_viols:.1f}")

        # Save checkpoint
        model.save(checkpoint_path)
        model.save(os.path.join(args.model_dir, f"ppo_alns_batch{batch_num:02d}"))
        print(f"  Checkpoint saved -> models/")

        # Save log and plot
        log_path = os.path.join(args.log_dir, "training_log.json")
        for i, ep in enumerate(episode_log):
            ep["episode"] = i
        with open(log_path, "w") as f:
            json.dump(episode_log, f)

        plot_training_log(log_path, batch_label=f"_batch{batch_num:02d}")

        batch_summaries.append({
            "batch":            batch_num,
            "avg_reward":       round(avg_reward, 4),
            "avg_improvement":  round(avg_improv, 2),
            "avg_violations":   round(avg_viols, 2),
            "elapsed_min":      round(elapsed, 1),
            "episodes":         len(new_eps),
        })
        with open(os.path.join(args.log_dir, "batch_summaries.json"), "w") as f:
            json.dump(batch_summaries, f, indent=2)

        plot_batch_summary(batch_summaries)
        print(f"  Charts saved -> outputs/")
        print(f"\n  You can stop now (Ctrl+C) or wait for next batch.")

        if batch_num < batch_start + total - 1:
            print(f"  [Next batch starts in 5 sec... Ctrl+C to stop]")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                print("\n  Stopped. Run with --resume to continue.")
                break

    final = os.path.join(args.model_dir, "ppo_alns_final")
    model.save(final)
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final model    -> {final}.zip")
    print(f"  Total episodes : {len(episode_log)}")
    print(f"{'='*60}\n")

    # ADD THESE TWO LINES
    print_training_summary(batch_summaries)
    plot_batch_summary(batch_summaries)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument("--data_dir",   default="data/synthetic_dataset")
    # p.add_argument("--data_dir", default="data/clean_dataset_v3")
    # p.add_argument("--model_dir",  default="models")
    # p.add_argument("--log_dir",    default="logs")
    # p.add_argument("--batches",    type=int, default=6)
    # p.add_argument("--max_files",  type=int, default=10)
    # p.add_argument("--max_iter",   type=int, default=20)
    # p.add_argument("--max_iter",   type=int, default=5)
    p.add_argument("--data_dir", default="data/dataset_v3")
    # p.add_argument("--data_dir", default="data/dataset_v4")
    p.add_argument("--model_dir",  default="models")
    p.add_argument("--log_dir",    default="logs")
    p.add_argument("--batches",    type=int, default=3)
    p.add_argument("--max_files",  type=int, default=100)
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--resume",     action="store_true")
    args = p.parse_args()
    train_batch(args)
    
