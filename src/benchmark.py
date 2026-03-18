"""
benchmark.py
============
Runs Greedy ALNS vs PPO-ALNS across all instances N times each,
then prints a comparison table suitable for a report.

Usage:
    python src/benchmark.py --data_dir data/clean_dataset_v2 --runs 25
"""

import argparse
import os
import sys
import time
import json
import random
from statistics import mean, stdev

# ── path fix so imports work from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_loader import build_dataset, augment_instance
from alns_env    import ALNSEnv, build_tw_sorted_route, Solution, W_LATENESS
from constraints import check_route, compute_metrics

try:
    from stable_baselines3 import PPO
    HAS_PPO = True
except ImportError:
    HAS_PPO = False


# ─────────────────────────────────────────────────────────────────────────────
# Single run — one instance, one mode
# ─────────────────────────────────────────────────────────────────────────────

def run_once(instance, mode: str, ppo_model, max_iter: int, seed: int) -> dict:
    rng = random.Random(seed)
    inst = augment_instance(instance, rng)
    orders_map = {o.order_id: o for o in inst.orders}

    # Build initial route
    from alns_env import build_tw_sorted_route
    # route = build_tw_sorted_route(inst.orders, orders_map)
    # current = Solution(route, orders_map, inst.vehicle)
    # best    = current.copy()
    route = build_tw_sorted_route(inst.orders, orders_map)
    current = Solution(route, orders_map, inst.vehicle)
    best    = current.copy()
    init_cost = max(current.cost(), 1.0)

    from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
    import numpy as np

    destroy_usage = np.zeros(N_DESTROY)
    repair_usage  = np.zeros(N_REPAIR)

    t0 = time.perf_counter()

    for iteration in range(max_iter):
        # Choose operator
        if mode == "ppo" and ppo_model is not None:
            # Build state
            it = iteration
            sp = it / max(max_iter, 1)
            curr_cost = current.cost()
            best_cost = best.cost()
            from alns_env import MAX_COST, MAX_TW, MAX_SERVICE, MAX_FUEL_COST
            #delta   = (curr_cost - best_cost) / max(current.cost(), 1.0)
            #init_cost = current.cost() if iteration == 0 else init_cost
            delta   = (curr_cost - best_cost) / max(init_cost, 1.0)
            total   = max(it, 1)
            d_usage = destroy_usage / total
            r_usage = repair_usage  / total
            tws = [(o.pickup_node.due_time - o.pickup_node.ready_time) for o in inst.orders]
            avg_tw  = np.mean(tws) / MAX_TW
            svc = [o.pickup_node.service_time + o.delivery_node.service_time for o in inst.orders]
            avg_svc = np.mean(svc) / MAX_SERVICE
            fck_norm = inst.vehicle.fuel_cost_per_km / MAX_FUEL_COST
            vt_norm  = inst.vehicle.type_index / 3.0
            n_orders = max(len(inst.orders), 1)
            viol_rate = best.violations() / n_orders
            state = np.array([
                sp,
                np.clip(delta, -1.0, 1.0),
                np.clip(current.cost() / MAX_COST, 0, 1),
                np.clip(best_cost / MAX_COST, 0, 1),
                *np.clip(d_usage, 0.0, 1.0),
                *np.clip(r_usage, 0.0, 1.0),
                np.clip(avg_tw,    0.0, 1.0),
                np.clip(avg_svc,   0.0, 1.0),
                np.clip(fck_norm,  0.0, 1.0),
                np.clip(vt_norm,   0.0, 1.0),
                np.clip(viol_rate, 0.0, 1.0),
            ], dtype=np.float32)
            action, _ = ppo_model.predict(state, deterministic=True)
            d_idx = int(action) // N_REPAIR
            r_idx = int(action) % N_REPAIR
        else:
            # Greedy: random operator selection
            d_idx = rng.randint(0, N_DESTROY - 1)
            r_idx = rng.randint(0, N_REPAIR  - 1)

        d_idx = min(d_idx, N_DESTROY - 1)
        r_idx = min(r_idx, N_REPAIR  - 1)

        destroy_op = DESTROY_OPS[d_idx]
        repair_op  = REPAIR_OPS[r_idx]

        n_remove = rng.randint(1, min(6, len(inst.orders)))
        prev_cost = current.cost()

        new_route, removed = destroy_op(current.route, orders_map, inst.vehicle, n_remove, rng)
        new_route = repair_op(new_route, removed, orders_map, inst.vehicle, rng)

        # candidate = Solution(new_route, orders_map, inst.vehicle)
        # if candidate.cost() < prev_cost:
        #     current = candidate
        # if candidate.cost() < best.cost():
        #     best = candidate.copy()
        import math
        candidate = Solution(new_route, orders_map, inst.vehicle)
        new_cost  = candidate.cost()

        progress    = iteration / max(max_iter, 1)
        # temperature = 10000.0 * (1.0 - progress) + 50.0
        temperature = 500.0 * (1.0 - progress) + 5.0

        if new_cost < prev_cost:
            current = candidate
        else:
            sa_prob = math.exp(-(new_cost - prev_cost) / max(temperature, 1e-9))
            if rng.random() < sa_prob:
                current = candidate

        if new_cost < best.cost():
            best = candidate.copy()

        destroy_usage[d_idx] += 1
        repair_usage[r_idx]  += 1

    elapsed = time.perf_counter() - t0

    result   = check_route(best.route, orders_map, inst.vehicle)
    metrics  = compute_metrics(best.route, inst.vehicle)

    return {
        "mode":            mode,
        "instance_id":     inst.instance_id,
        "num_orders":      len(inst.orders),
        "violations":      result.deadline_violations,
        "total_lateness":  round(result.total_lateness, 2),
        "distance_km":     round(metrics["total_distance_km"], 4),
        "travel_time_min": round(metrics["travel_time_min"], 2),
        "fuel_cost_inr":   round(metrics["fuel_cost_inr"], 2),
        "carbon_kg":       round(metrics["carbon_emission_kg"], 4),
        "objective_cost":  round(best.cost(), 2),
        "elapsed_sec":     round(elapsed, 3),
        "feasible":        result.feasible,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir",  default="data/clean_dataset_v2")
    parser.add_argument("--data_dir", default="data/clean_dataset_v3")
    parser.add_argument("--runs",      type=int, default=25,
                        help="Number of runs per instance per mode")
    parser.add_argument("--max_iter",  type=int, default=50)
    parser.add_argument("--model_dir", default="models",
                        help="Directory containing ppo_alns_batch*.zip files")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--max_files", type=int, default=250)
    
    args = parser.parse_args()

    instances_dir = os.path.join(args.data_dir, "instances")
    #instances = build_dataset(instances_dir, max_files=50)
    instances = build_dataset(instances_dir, max_files=args.max_files if hasattr(args, 'max_files') else 250)
    if not instances:
        print("No instances loaded. Check --data_dir.")
        sys.exit(1)

    # Load latest PPO model
    ppo_model = None
    if HAS_PPO:
        zips = sorted([
            f for f in os.listdir(args.model_dir)
            if f.startswith("ppo_alns_batch") and f.endswith(".zip")
        ])
        if zips:
            model_path = os.path.join(args.model_dir, zips[-1])
            print(f"Loading PPO model: {model_path}")
            ppo_model = PPO.load(model_path)
        else:
            print("No PPO model found in models/ — PPO mode will use random operators.")

    print(f"\nBenchmark: {len(instances)} instances × {args.runs} runs × 2 modes")
    print(f"Max iterations per run: {args.max_iter}\n")

    greedy_results = []
    ppo_results    = []

    total_runs = len(instances) * args.runs * 2
    done = 0

    for inst in instances:
        for run in range(args.runs):
            seed = args.seed + run * 1000 + hash(inst.instance_id) % 1000

            g = run_once(inst, "greedy", None,      args.max_iter, seed)
            p = run_once(inst, "ppo",    ppo_model, args.max_iter, seed)

            greedy_results.append(g)
            ppo_results.append(p)
            done += 2

            print(f"  [{done:4d}/{total_runs}] {inst.instance_id:25s} "
                  f"run={run+1:2d}  "
                  f"greedy_viol={g['violations']:3d}  ppo_viol={p['violations']:3d}  "
                  f"greedy_cost={g['objective_cost']:9.1f}  ppo_cost={p['objective_cost']:9.1f}")

    # ── Aggregate ──
    def avg(lst, key): return mean(r[key] for r in lst)
    def sd(lst, key):
        vals = [r[key] for r in lst]
        return stdev(vals) if len(vals) > 1 else 0.0

    metrics_keys = [
        ("violations",      "Deadline Violations",  ".2f"),
        ("total_lateness",  "Total Lateness (min)", ".1f"),
        ("distance_km",     "Distance (km)",        ".3f"),
        ("travel_time_min", "Travel Time (min)",    ".2f"),
        ("fuel_cost_inr",   "Fuel Cost (INR)",      ".2f"),
        ("carbon_kg",       "Carbon (kg CO2)",      ".4f"),
        ("objective_cost",  "Objective Cost",       ".2f"),
        ("elapsed_sec",     "Runtime (sec)",        ".3f"),
    ]

    print("\n" + "="*72)
    print("  BENCHMARK RESULTS")
    print(f"  Instances: {len(instances)}  |  Runs per instance: {args.runs}  |  Total runs: {len(greedy_results)}")
    print("="*72)
    print(f"  {'Metric':<26}  {'Greedy':>12}  {'PPO':>12}  {'Improvement':>12}")
    print("-"*72)

    summary = {}
    for key, label, fmt in metrics_keys:
        g_avg = avg(greedy_results, key)
        p_avg = avg(ppo_results,    key)
        if g_avg != 0:
            imp = (g_avg - p_avg) / abs(g_avg) * 100
        else:
            imp = 0.0
        direction = "▼" if p_avg < g_avg else ("▲" if p_avg > g_avg else "=")
        print(f"  {label:<26}  {g_avg:>12{fmt}}  {p_avg:>12{fmt}}  {direction}{abs(imp):>10.1f}%")
        summary[key] = {"greedy": g_avg, "ppo": p_avg, "improvement_pct": round(imp, 2)}

    print("="*72)

    # Feasibility
    g_feas = sum(1 for r in greedy_results if r["feasible"]) / len(greedy_results) * 100
    p_feas = sum(1 for r in ppo_results    if r["feasible"]) / len(ppo_results)    * 100
    print(f"  {'Feasible Routes':<26}  {g_feas:>11.1f}%  {p_feas:>11.1f}%")
    print("="*72)

    # Per-instance summary
    print("\n  PER-INSTANCE AVERAGE VIOLATIONS (Greedy vs PPO)")
    print(f"  {'Instance':<28} {'Orders':>6}  {'Greedy Viol':>11}  {'PPO Viol':>8}  {'Delta':>6}")
    print("  " + "-"*65)
    for inst in instances:
        g_inst = [r for r in greedy_results if r["instance_id"] == inst.instance_id]
        p_inst = [r for r in ppo_results    if r["instance_id"] == inst.instance_id]
        if g_inst and p_inst:
            gv = mean(r["violations"] for r in g_inst)
            pv = mean(r["violations"] for r in p_inst)
            print(f"  {inst.instance_id:<28} {len(inst.orders):>6}  {gv:>11.2f}  {pv:>8.2f}  {gv-pv:>+6.2f}")

    # Save JSON
    out = {
        "config": vars(args),
        "summary": summary,
        "feasibility": {"greedy_pct": g_feas, "ppo_pct": p_feas},
        "greedy_results": greedy_results,
        "ppo_results":    ppo_results,
    }
    out_path = "outputs/benchmark_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    main()
