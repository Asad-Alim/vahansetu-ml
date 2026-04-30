

"""
presolve.py
===========
OR-Tools warm-start pre-solver for PPO-ALNS.

Run this ONCE before training to generate a cache of high-quality
feasible initial solutions for every instance in your dataset.

Usage:
    python src/presolve.py
    python src/presolve.py --data_dir data/clean_dataset_v3 --cache_dir data/or_cache
    python src/presolve.py --timeout 10   # seconds per instance (default: 5)

Output:
    data/or_cache/<instance_id>.json   — one file per instance

Each cache file contains:
    {
        "instance_id": "...",
        "route": [
            {"order_id": "...", "node_id": "...", "is_pickup": true/false},
            ...
        ],
        "cost": 123.45,
        "solver_status": "OPTIMAL" | "FEASIBLE" | "GREEDY_FALLBACK"
    }

How alns_env.py uses this:
    In reset(), instead of calling build_tw_sorted_route(), it loads the
    cached route for the chosen instance and uses it as the starting point.
    If no cache file exists for an instance, it falls back to the original
    greedy initializer automatically.
"""

import os
import sys
import json
import math
import time
import argparse
import random
from typing import List, Dict, Optional, Tuple

def greedy_tw_route(orders, orders_map):
    """Wrapper — delegates to the superior build_tw_sorted_route from alns_env."""
    from alns_env import build_tw_sorted_route  # lazy import to avoid circular dependency
    return build_tw_sorted_route(orders, orders_map)

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    build_dataset, Instance, Order, Node, VehicleConfig,
    get_travel_time, get_distance, reset_travel_cache,
    augment_instance, VEHICLE_CATALOGUE
)
from constraints import RouteNode, check_route, compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Cost function — mirrors alns_env.py exactly so cached cost is comparable
# ─────────────────────────────────────────────────────────────────────────────

# W_TRAVEL_TIME = 1.0
# W_LATENESS    = 10.0
# W_CARBON      = 0.05
# W_FUEL        = 0.1
# W_INFEASIBLE  = 1e6
from constants import W_TRAVEL_TIME, W_LATENESS, W_CARBON, W_FUEL, W_INFEASIBLE

def route_cost(route: List[RouteNode],
               orders_map: Dict[str, Order],
               vehicle: VehicleConfig) -> float:
    result  = check_route(route, orders_map, vehicle)
    metrics = compute_metrics(route, vehicle)
    c = (W_TRAVEL_TIME * metrics["travel_time_min"]
       + W_LATENESS    * result.total_lateness
       + W_CARBON      * metrics["carbon_emission_kg"]
       + W_FUEL        * metrics["fuel_cost_inr"])
    if not result.feasible:
        c += W_INFEASIBLE
    return round(c, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy fallback — time-window sorted insertion (same as alns_env.py)
# ─────────────────────────────────────────────────────────────────────────────

# def greedy_tw_route(orders: List[Order],
#                     orders_map: Dict[str, Order]) -> List[RouteNode]:
#     """
#     Earliest-deadline-first greedy insertion.
#     Mirrors build_tw_sorted_route() from alns_env.py.
#     Used as fallback when OR-Tools fails or is not installed.
#     """
#     sorted_orders = sorted(orders, key=lambda o: o.delivery_node.due_time)
#     route = []
#     for o in sorted_orders:
#         route.append(RouteNode(order_id=o.order_id, node=o.pickup_node))
#         route.append(RouteNode(order_id=o.order_id, node=o.delivery_node))
#     return route


# ─────────────────────────────────────────────────────────────────────────────
# OR-Tools solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_with_ortools(instance: Instance,
                       timeout_seconds: int = 5) -> Tuple[List[RouteNode], str]:
    """
    Solve the PDVRPTW instance using OR-Tools CP-SAT / Routing solver.

    Returns:
        (route, status_string)
        status is one of: "OPTIMAL", "FEASIBLE", "GREEDY_FALLBACK"
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2
        #from ortools.constraint_solver import pywrapcrouting as pywrapcp
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        print("  [WARN] OR-Tools not installed. Using greedy fallback.")
        print("         Install with: pip install ortools")
        route = greedy_tw_route(instance.orders, {o.order_id: o for o in instance.orders})
        return route, "GREEDY_FALLBACK"

    orders_map = {o.order_id: o for o in instance.orders}

    # ── Build flat node list: depot(0), then pickup/delivery pairs ──
    # Node layout: 0=depot, 1=P0, 2=D0, 3=P1, 4=D1, ...
    nodes: List[Node] = [instance.depot]
    for o in instance.orders:
        nodes.append(o.pickup_node)
        nodes.append(o.delivery_node)

    n = len(nodes)  # total nodes including depot

    # ── Travel time matrix (integer minutes, OR-Tools requires int) ──
    def travel_min(i: int, j: int) -> int:
        if i == j:
            return 0
        return max(1, int(get_travel_time(nodes[i], nodes[j])))

    time_matrix = [[travel_min(i, j) for j in range(n)] for i in range(n)]

    # ── OR-Tools routing model ──
    # Single vehicle: 1 vehicle, depot at index 0
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback
    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        service = nodes[i].service_time if i != 0 else 0.0
        return time_matrix[i][j] + int(service)

    transit_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Time window dimension
    MAX_TIME = 1440  # full day in minutes
    routing.AddDimension(
        transit_cb,
        slack_max=MAX_TIME,   # waiting time allowed
        capacity=MAX_TIME,
        fix_start_cumul_to_zero=True,
        name="Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Apply time windows
    for node_idx, node in enumerate(nodes):
        if node_idx == 0:
            continue  # depot — no TW constraint
        index = manager.NodeToIndex(node_idx)
        time_dim.CumulVar(index).SetRange(
            int(node.ready_time),
            int(node.due_time)
        )

    # Pickup-delivery precedence constraints
    for i, o in enumerate(instance.orders):
        p_node_idx = 1 + i * 2      # pickup index in nodes list
        d_node_idx = 1 + i * 2 + 1  # delivery index in nodes list
        p_index = manager.NodeToIndex(p_node_idx)
        d_index = manager.NodeToIndex(d_node_idx)
        routing.AddPickupAndDelivery(p_index, d_index)
        routing.solver().Add(
            routing.VehicleVar(p_index) == routing.VehicleVar(d_index)
        )
        routing.solver().Add(
            time_dim.CumulVar(p_index) <= time_dim.CumulVar(d_index)
        )

    # Capacity constraint (weight)
    def demand_callback(from_idx):
        node_idx = manager.IndexToNode(from_idx)
        if node_idx == 0:
            return 0
        # odd node_idx = pickup (+weight), even = delivery (-weight)
        order_i = (node_idx - 1) // 2
        order   = instance.orders[order_i]
        if nodes[node_idx].is_pickup:
            return int(order.weight_kg)
        else:
            return -int(order.weight_kg)

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb,
        slack_max=0,
        vehicle_capacities=[int(instance.vehicle.max_weight)],
        fix_start_cumul_to_zero=True,
        name="Capacity"
    )

    # ── Search parameters ──
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = timeout_seconds
    search_params.log_search = False

    # ── Solve ──
    solution = routing.SolveWithParameters(search_params)

    if solution is None:
        print("  [WARN] OR-Tools found no solution. Using greedy fallback.")
        route = greedy_tw_route(instance.orders, orders_map)
        return route, "GREEDY_FALLBACK"

    # ── Extract route from OR-Tools solution ──
    route: List[RouteNode] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        if node_idx != 0:  # skip depot visits
            node    = nodes[node_idx]
            order_i = (node_idx - 1) // 2
            order   = instance.orders[order_i]
            route.append(RouteNode(order_id=order.order_id, node=node))
        index = solution.Value(routing.NextVar(index))

    if not route:
        print("  [WARN] OR-Tools returned empty route. Using greedy fallback.")
        route = greedy_tw_route(instance.orders, orders_map)
        return route, "GREEDY_FALLBACK"

    status = "OPTIMAL" if routing.status() == 1 else "FEASIBLE"
    return route, status


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def route_to_json(route: List[RouteNode]) -> List[Dict]:
    """Convert a route to a JSON-serialisable list of dicts."""
    return [
        {
            "order_id":  rn.order_id,
            "node_id":   rn.node.node_id,
            "is_pickup": rn.node.is_pickup
        }
        for rn in route
    ]


def route_from_json(data: List[Dict],
                    orders_map: Dict[str, Order]) -> Optional[List[RouteNode]]:
    """
    Reconstruct a route from cached JSON.
    Returns None if any order_id or node_id is missing (instance changed).
    """
    route = []
    for entry in data:
        oid      = entry["order_id"]
        node_id  = entry["node_id"]
        is_pickup = entry["is_pickup"]

        if oid not in orders_map:
            return None  # cache is stale

        order = orders_map[oid]
        node  = order.pickup_node if is_pickup else order.delivery_node

        if node.node_id != node_id:
            return None  # node mismatch — cache is stale

        route.append(RouteNode(order_id=oid, node=node))

    return route if route else None


# ─────────────────────────────────────────────────────────────────────────────
# Main pre-solver loop
# ─────────────────────────────────────────────────────────────────────────────

def presolve_all(data_dir: str,
                 cache_dir: str,
                 max_files: int,
                 timeout: int,
                 force: bool) -> None:

    instances_dir = os.path.join(data_dir, "instances")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PPO-ALNS OR-Tools Pre-Solver")
    print(f"{'='*60}")
    print(f"  Instances dir : {instances_dir}")
    print(f"  Cache dir     : {cache_dir}")
    print(f"  Timeout/inst  : {timeout}s")
    print(f"  Max files     : {max_files}")
    print(f"  Force rerun   : {force}")
    print(f"{'='*60}\n")

    instances = build_dataset(instances_dir, max_files=max_files)
    if not instances:
        print("ERROR: No instances found. Check --data_dir path.")
        return

    stats = {"optimal": 0, "feasible": 0, "fallback": 0, "skipped": 0}
    total_start = time.time()

    for i, inst in enumerate(instances, 1):
        cache_path = os.path.join(cache_dir, f"{inst.instance_id}.json")

        # Skip if already cached and not forcing rerun
        if os.path.exists(cache_path) and not force:
            print(f"  [{i:3d}/{len(instances)}] {inst.instance_id} — SKIPPED (cached)")
            stats["skipped"] += 1
            continue

        print(f"  [{i:3d}/{len(instances)}] {inst.instance_id} "
              f"({len(inst.orders)} orders) ... ", end="", flush=True)

        # Augment instance with a fixed vehicle for consistent caching
        # Use the most capable vehicle to maximise feasibility
        rng = random.Random(42)
        augment_instance(inst, rng)

        orders_map = {o.order_id: o for o in inst.orders}

        t0 = time.time()
        route, status = solve_with_ortools(inst, timeout_seconds=timeout)
        elapsed = time.time() - t0

        cost = route_cost(route, orders_map, inst.vehicle)

        # Save to cache
        cache_entry = {
            "instance_id":   inst.instance_id,
            "n_orders":      len(inst.orders),
            "solver_status": status,
            "cost":          cost,
            "elapsed_sec":   round(elapsed, 2),
            "route":         route_to_json(route)
        }
        with open(cache_path, "w") as f:
            json.dump(cache_entry, f, indent=2)

        print(f"{status} | cost={cost:.1f} | {elapsed:.1f}s")

        if status == "OPTIMAL":
            stats["optimal"] += 1
        elif status == "FEASIBLE":
            stats["feasible"] += 1
        else:
            stats["fallback"] += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pre-solve complete in {total_elapsed:.1f}s")
    print(f"  Optimal        : {stats['optimal']}")
    print(f"  Feasible       : {stats['feasible']}")
    print(f"  Greedy fallback: {stats['fallback']}")
    print(f"  Skipped(cached): {stats['skipped']}")
    print(f"  Cache dir      : {cache_dir}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pre-solve VRPTW instances with OR-Tools and cache results."
    )
    p.add_argument("--data_dir",   default="data/clean_dataset_v3",
                   help="Root data directory (contains 'instances/' subfolder)")
    p.add_argument("--cache_dir",  default="data/or_cache",
                   help="Output directory for cached solutions")
    p.add_argument("--max_files",  type=int, default=200,
                   help="Max number of instance files to process")
    p.add_argument("--timeout",    type=int, default=5,
                   help="OR-Tools time limit per instance in seconds")
    p.add_argument("--force",      action="store_true",
                   help="Re-solve even if cache file already exists")
    args = p.parse_args()

    presolve_all(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        timeout=args.timeout,
        force=args.force,
    )