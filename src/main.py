



"""
main.py — Demo and inference runner for PPO-ALNS VRPTW.

Usage:
    python src/main.py                    # greedy demo
    python src/main.py --use_ppo          # with trained PPO model
    python src/main.py --seed 42          # fixed instance (repeatable demo)
    python src/main.py --start_time 09:00
"""

import os, sys, json, argparse, random, datetime
sys.path.insert(0, os.path.dirname(__file__))

from data_loader    import build_dataset, augment_instance, get_travel_time
from alns_env       import ALNSEnv, Solution, W_TRAVEL_TIME, W_LATENESS, W_CARBON, W_FUEL
from alns_env       import build_tw_sorted_route
from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
from constraints    import RouteNode, check_route, compute_metrics, format_schedule


# -----------------------------------------------------------------------------
# Greedy ALNS baseline — uses same TW-sorted initial route as PPO
# -----------------------------------------------------------------------------

def run_alns_greedy(instance, max_iter=50, seed=42):
    """
    Pure ALNS with random operator selection (no PPO).
    Uses time-window sorted greedy insertion as initial solution
    so the baseline is fair and starts from a reasonable state.
    """
    rng        = random.Random(seed)
    orders_map = {o.order_id: o for o in instance.orders}

    # Time-window sorted initial route (same as PPO env)
    route   = build_tw_sorted_route(instance.orders, orders_map)
    current = Solution(route, orders_map, instance.vehicle)
    best    = current.copy()

    for _ in range(max_iter):
        d_idx    = rng.randint(0, N_DESTROY - 1)
        r_idx    = rng.randint(0, N_REPAIR  - 1)
        n_remove = rng.randint(1, min(5, len(instance.orders)))

        new_route, removed = DESTROY_OPS[d_idx](
            current.route, orders_map, instance.vehicle, n_remove, rng)
        new_route = REPAIR_OPS[r_idx](
            new_route, removed, orders_map, instance.vehicle, rng)

        candidate = Solution(new_route, orders_map, instance.vehicle)
        if candidate.cost() < current.cost():
            current = candidate
            if candidate.cost() < best.cost():
                best = candidate.copy()

    return best


# -----------------------------------------------------------------------------
# PPO-ALNS
# -----------------------------------------------------------------------------

def run_ppo_alns(instance, model_path, max_iter=50):
    from stable_baselines3 import PPO as SB3PPO
    try:
        model = SB3PPO.load(model_path)
    except Exception as e:
        print(f"[WARN] Could not load PPO model: {e}\nFalling back to greedy.")
        return run_alns_greedy(instance, max_iter)

    env = ALNSEnv([instance], max_iter=max_iter)
    obs, _ = env.reset()
    for _ in range(max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            break
    return env.best


# -----------------------------------------------------------------------------
# Output formatter
# -----------------------------------------------------------------------------

def print_results_with_viz(instance, solution, start_time="08:00"):
    orders_map = {o.order_id: o for o in instance.orders}
    result  = check_route(solution.route, orders_map, instance.vehicle)
    metrics = compute_metrics(solution.route, instance.vehicle)
    sched   = format_schedule(solution.route, result.arrival_times,
                               result.departure_times, start_time)
    for i, rn in enumerate(solution.route):
        sched[i]["lat"] = rn.node.lat
        sched[i]["lon"] = rn.node.lon

    print(f"\n{'='*82}")
    print(f"  INSTANCE : {instance.instance_id}")
    print(f"  VEHICLE  : {instance.vehicle.vehicle_type} | "
          f"{instance.vehicle.fuel_type} | "
          f"{instance.vehicle.mileage} km/l | "
          f"Rs{instance.vehicle.fuel_price}/l | "
          f"{instance.vehicle.emission_factor} kg CO2/km")
    print(f"  CAPACITY : {instance.vehicle.max_weight} kg | {instance.vehicle.max_volume} m3")
    print(f"  ORDERS   : {len(instance.orders)}")
    print(f"{'='*82}")

    print(f"\n{'#':<4} {'Type':<10} {'District':<22} {'Lat':>9} {'Lon':>9} "
          f"{'ETA':>6} {'ETD':>6} {'Svc':>5}")
    print("-"*82)
    for i, s in enumerate(sched):
        print(f"{i+1:<4} {s['type']:<10} {s['district']:<22} "
              f"{s['lat']:>9.5f} {s['lon']:>9.5f} "
              f"{s['eta']:>6} {s['etd']:>6} "
              f"{s['service_time_min']:>5.0f}m")

    weight_util = round(
        sum(o.weight_kg for o in instance.orders) / instance.vehicle.max_weight * 100, 1)

    print(f"\n--- SUMMARY ---")
    print(f"  Orders              : {len(instance.orders)}")
    print(f"  Stops               : {len(solution.route)}")
    print(f"  Districts visited   : {len(set(s['district'] for s in sched))}")
    print(f"  Total distance      : {metrics['total_distance_km']} km")
    print(f"  Travel time         : {metrics['travel_time_min']} min")
    print(f"  Carbon emission     : {metrics['carbon_emission_kg']} kg CO2")
    print(f"  Fuel cost           : Rs{metrics['fuel_cost_inr']:.2f}")
    print(f"  Weight utilisation  : {weight_util}%")
    print(f"  Deadline violations : {result.deadline_violations}")
    print(f"  Total lateness      : {result.total_lateness:.1f} min")
    print(f"  Feasible            : {'YES' if result.feasible else 'NO'}")
    print(f"  Objective cost      : {solution.cost():.2f}")

    output = {
        "instance_id":      instance.instance_id,
        "generated_at":     datetime.datetime.now().isoformat(),
        "route_start_time": start_time,
        "region":           "Delhi NCR",
        "vehicle": {
            "type":             instance.vehicle.vehicle_type,
            "fuel":             instance.vehicle.fuel_type,
            "mileage_kmpl":     instance.vehicle.mileage,
            "fuel_price_inr":   instance.vehicle.fuel_price,
            "carbon_per_km":    instance.vehicle.emission_factor,
            "fuel_cost_per_km": instance.vehicle.fuel_cost_per_km,
            "max_weight_kg":    instance.vehicle.max_weight,
            "max_volume_m3":    instance.vehicle.max_volume,
        },
        "route":             sched,
        "districts_visited": sorted(set(s["district"] for s in sched)),
        "summary": {
            **metrics,
            "num_orders":             len(instance.orders),
            "num_stops":              len(solution.route),
            "deadline_violations":    result.deadline_violations,
            "total_lateness_min":     result.total_lateness,
            "feasible":               result.feasible,
            "objective_cost":         solution.cost(),
            "weight_utilisation_pct": weight_util,
        }
    }

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{instance.instance_id}_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON saved  -> {out_path}")

    try:
        from visualizer import plot_route, plot_cost_breakdown
        plot_route(instance, solution)
        plot_cost_breakdown(metrics, result.total_lateness, instance.instance_id)
        print(f"  Charts saved -> outputs/")
    except Exception as e:
        print(f"  [Charts skipped: {e}]")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/synthetic_dataset")
    p.add_argument("--use_ppo",    action="store_true")
    p.add_argument("--model_path", default="models/ppo_alns_final")
    p.add_argument("--max_iter",   type=int, default=50)
    p.add_argument("--start_time", default="08:00")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    instances = build_dataset(
        os.path.join(args.data_dir, "instances"), max_files=10)
    if not instances:
        print("No instances loaded. Check --data_dir")
        sys.exit(1)

    rng  = random.Random(args.seed)
    inst = rng.choice(instances)
    inst = augment_instance(inst, rng)
    print(f"\nRunning on: {inst.instance_id} ({len(inst.orders)} orders)")

    if args.use_ppo:
        print("Mode: PPO-ALNS")
        solution = run_ppo_alns(inst, args.model_path, args.max_iter)
    else:
        print("Mode: Greedy ALNS (no PPO)")
        solution = run_alns_greedy(inst, args.max_iter, args.seed)

    print_results_with_viz(inst, solution, args.start_time)
