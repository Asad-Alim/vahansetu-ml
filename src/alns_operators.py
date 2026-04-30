

"""
alns_operators.py
=================
5 destroy operators + 3 repair operators for PPO-ALNS.
 
All operators share the same signature:
  destroy(route, orders_map, vehicle, n_remove, rng) -> (new_route, removed_ids)
  repair(route, removed_ids, orders_map, vehicle, rng) -> new_route
 
Performance notes:
  - _insertion_cost now uses a fast travel-time-only estimate instead of
    calling check_route (which does a full feasibility scan every call).
    check_route is O(n) and was being called O(n²) times per repair step.
  - repair_regret inner insertion loop is capped at MAX_POS positions.
  - repair_regret outer loop evaluates each remaining order once per round,
    not repeatedly.
"""
 
import math
import random
from typing import List, Tuple, Dict, Set
from data_loader import Order, VehicleConfig, get_travel_time
from constraints import RouteNode, check_route
 
 
# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
from constants import W_LATENESS
MAX_POS = 25        # max insertion positions tried per order (caps O(n²) loops)
# MAX_POS = 81
# W_LATENESS = 500.0   # must match alns_env.py so cost estimates are consistent
 
 
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
 
def _remove_order(route: List[RouteNode], order_id: str) -> List[RouteNode]:
    return [rn for rn in route if rn.order_id != order_id]
 
 
def _fast_travel_cost(route: List[RouteNode]) -> float:
    """
    Fast O(n) travel time estimate — no check_route call.
    Used for insertion cost comparisons where relative ordering matters,
    not absolute feasibility.
    """
    if len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += get_travel_time(route[i].node, route[i+1].node)
    return total
 
 
def _lateness_estimate(route: List[RouteNode], orders_map: Dict) -> float:
    """
    Fast O(n) lateness estimate using simple cumulative time tracking.
    Much faster than check_route — no full feasibility scan.
    """
    if not route:
        return 0.0
    time_now = 0.0
    lateness = 0.0
    prev_node = None
    for rn in route:
        node = rn.node
        if prev_node is not None:
            time_now += get_travel_time(prev_node, node)
        # Wait for ready time if early
        time_now = max(time_now, node.ready_time)
        # Lateness if past due time
        if time_now > node.due_time:
            lateness += time_now - node.due_time
        time_now += node.service_time
        prev_node = node
    return lateness
 
 
# def _insertion_cost_fast(route: List[RouteNode],
#                          order: Order,
#                          pos_pickup: int,
#                          pos_delivery: int,
#                          orders_map: Dict) -> float:
#     """
#     Fast insertion cost estimate — O(n) travel + O(n) lateness.
#     No check_route call. Used for comparing insertion positions.
#     """
#     trial = list(route)
#     trial.insert(pos_pickup,   RouteNode(order.order_id, order.pickup_node))
#     trial.insert(pos_delivery, RouteNode(order.order_id, order.delivery_node))
#     travel  = _fast_travel_cost(trial)
#     late    = _lateness_estimate(trial, orders_map)
#     return travel + W_LATENESS * late
def _insertion_cost_fast(route: List[RouteNode],
                         order: Order,
                         pos_pickup: int,
                         pos_delivery: int,
                         orders_map: Dict,
                         vehicle=None) -> float:
    """
    Fast insertion cost estimate — O(n) travel + O(n) lateness.
    If vehicle is provided, rejects positions that violate weight capacity
    by returning float('inf'), so the solver never picks an overweight route.
    """
    trial = list(route)
    trial.insert(pos_pickup,   RouteNode(order.order_id, order.pickup_node))
    trial.insert(pos_delivery, RouteNode(order.order_id, order.delivery_node))

    # Weight feasibility check — PDPTW: weight at any moment must not exceed max
    # if vehicle is not None:
    #     curr_w = 0.0
    #     for rn in trial:
    #         o = orders_map[rn.order_id]
    #         curr_w += o.weight_kg if rn.is_pickup else -o.weight_kg
    #         if curr_w > vehicle.max_weight + 1e-6:
    #             return float('inf')  # hard reject this position

    if vehicle is not None:
        curr_w = 0.0
        curr_v = 0.0
        for rn in trial:
            o = orders_map[rn.order_id]
            if rn.is_pickup:
                curr_w += o.weight_kg
                curr_v += o.volume_m3
            else:
                curr_w -= o.weight_kg
                curr_v -= o.volume_m3
            if curr_w > vehicle.max_weight + 1e-6:
                return float('inf')
            if curr_v > vehicle.max_volume + 1e-6:
                return float('inf')        

    travel = _fast_travel_cost(trial)
    late   = _lateness_estimate(trial, orders_map)
    return travel + W_LATENESS * late
 
 
# def _best_insertion(route: List[RouteNode],
#                     order: Order,
#                     orders_map: Dict,
#                     max_pos: int = MAX_POS) -> Tuple[float, List[RouteNode]]:
#     """
#     Find best insertion position for one order.
#     Returns (best_cost, best_route).
#     Caps search at max_pos to keep it O(max_pos²) not O(n²).
#     """
#     n = len(route)
#     limit = min(n + 1, max_pos)
#     best_cost  = float('inf')
#     best_route = None
 
#     for i in range(limit):
#         for j in range(i + 1, min(limit + 1, n + 2)):
#             cost = _insertion_cost_fast(route, order, i, j, orders_map)
#             if cost < best_cost:
#                 best_cost = cost
#                 r = list(route)
#                 r.insert(i, RouteNode(order.order_id, order.pickup_node))
#                 r.insert(j, RouteNode(order.order_id, order.delivery_node))
#                 best_route = r
 
#     if best_route is None:
#         # Fallback: append at end
#         best_route = route + [
#             RouteNode(order.order_id, order.pickup_node),
#             RouteNode(order.order_id, order.delivery_node),
#         ]
#         best_cost = _fast_travel_cost(best_route)
 
#     return best_cost, best_route

# def _best_insertion(route: List[RouteNode],
#                     order: Order,
#                     orders_map: Dict,
#                     max_pos: int = MAX_POS,
#                     vehicle=None) -> Tuple[float, List[RouteNode]]:
#     """
#     Find best insertion position for one order.
#     Returns (best_cost, best_route).
#     Caps search at max_pos to keep it O(max_pos²) not O(n²).
#     Passes vehicle to _insertion_cost_fast so weight violations are hard-rejected.
#     """
#     n = len(route)
#     limit = min(n + 1, max_pos)
#     best_cost  = float('inf')
#     best_route = None

#     for i in range(limit):
#         for j in range(i + 1, min(limit + 1, n + 2)):
#             cost = _insertion_cost_fast(route, order, i, j, orders_map, vehicle)
#             if cost < best_cost:
#                 best_cost = cost
#                 r = list(route)
#                 r.insert(i, RouteNode(order.order_id, order.pickup_node))
#                 r.insert(j, RouteNode(order.order_id, order.delivery_node))
#                 best_route = r

#     if best_route is None:
#         # All positions violated weight — fallback to end (feasibility will
#         # be caught by check_route and W_INFEASIBLE penalty applied)
#         best_route = route + [
#             RouteNode(order.order_id, order.pickup_node),
#             RouteNode(order.order_id, order.delivery_node),
#         ]
#         best_cost = _fast_travel_cost(best_route)

#     return best_cost, best_route
 # In alns_operators.py, change _best_insertion:
def _best_insertion(route: List[RouteNode],
                    order: Order,
                    orders_map: Dict,
                    max_pos: int = None,   # None = no cap
                    vehicle=None) -> Tuple[float, List[RouteNode]]:
    n = len(route)
    limit = (n + 1) if max_pos is None else min(n + 1, max_pos)
    best_cost  = float('inf')
    best_route = None

    for i in range(limit):
        for j in range(i + 1, n + 2):   # delivery always searches full range
            cost = _insertion_cost_fast(route, order, i, j, orders_map, vehicle)
            if cost < best_cost:
                best_cost = cost
                r = list(route)
                r.insert(i, RouteNode(order.order_id, order.pickup_node))
                r.insert(j, RouteNode(order.order_id, order.delivery_node))
                best_route = r

    if best_route is None:
        best_route = route + [
            RouteNode(order.order_id, order.pickup_node),
            RouteNode(order.order_id, order.delivery_node),
        ]
        best_cost = _fast_travel_cost(best_route) + W_LATENESS * _lateness_estimate(best_route, orders_map)

    return best_cost, best_route
 
# -----------------------------------------------------------------------------
# Destroy operators
# -----------------------------------------------------------------------------
 
def destroy_random(route, orders_map, vehicle, n_remove, rng):
    """Remove n_remove randomly chosen orders."""
    order_ids = list({rn.order_id for rn in route})
    to_remove = rng.sample(order_ids, min(n_remove, len(order_ids)))
    new_route = [rn for rn in route if rn.order_id not in to_remove]
    return new_route, to_remove
 
 
def destroy_worst_cost(route, orders_map, vehicle, n_remove, rng):
    """
    Remove orders that contribute the most to total cost.
    Uses fast travel+lateness estimate instead of full check_route.
    """
    order_ids = list({rn.order_id for rn in route})
    base_cost = _fast_travel_cost(route) + W_LATENESS * _lateness_estimate(route, orders_map)
 
    savings = []
    for oid in order_ids:
        reduced = _remove_order(route, oid)
        if reduced:
            rc = _fast_travel_cost(reduced) + W_LATENESS * _lateness_estimate(reduced, orders_map)
        else:
            rc = 0.0
        savings.append((base_cost - rc, oid))
 
    savings.sort(reverse=True)
    to_remove = [oid for _, oid in savings[:n_remove]]
    new_route = [rn for rn in route if rn.order_id not in to_remove]
    return new_route, to_remove
 
 
def destroy_shaw(route, orders_map, vehicle, n_remove, rng):
    """
    Shaw removal: remove orders similar to a seed order
    (close in distance + time window).
    """
    order_ids = list({rn.order_id for rn in route})
    if not order_ids:
        return route, []
 
    seed_id    = rng.choice(order_ids)
    seed_order = orders_map[seed_id]
    seed_node  = seed_order.pickup_node
 
    def similarity(oid):
        o  = orders_map[oid]
        dx = seed_node.x - o.pickup_node.x
        dy = seed_node.y - o.pickup_node.y
        dist    = math.sqrt(dx*dx + dy*dy)
        tw_diff = abs(seed_order.pickup_node.due_time - o.pickup_node.due_time)
        return dist + 0.0025 * tw_diff
 
    scored    = sorted([(similarity(oid), oid) for oid in order_ids if oid != seed_id])
    to_remove = [seed_id] + [oid for _, oid in scored[:n_remove - 1]]
    new_route = [rn for rn in route if rn.order_id not in to_remove]
    return new_route, to_remove
 
 
def destroy_string(route, orders_map, vehicle, n_remove, rng):
    """Remove a contiguous string of stops from the route."""
    if len(route) <= n_remove:
        return destroy_random(route, orders_map, vehicle, n_remove, rng)
 
    start        = rng.randint(0, len(route) - n_remove)
    affected_ids = {route[i].order_id for i in range(start, start + n_remove)}
    new_route    = [rn for rn in route if rn.order_id not in affected_ids]
    return new_route, list(affected_ids)
 
 
def destroy_route_segment(route, orders_map, vehicle, n_remove, rng):
    """
    Remove orders in a geographic cluster
    (nodes close to a random center point).
    """
    if not route:
        return route, []
 
    cx = rng.uniform(min(rn.node.x for rn in route), max(rn.node.x for rn in route))
    cy = rng.uniform(min(rn.node.y for rn in route), max(rn.node.y for rn in route))
 
    def dist_to_center(oid):
        n = orders_map[oid].pickup_node
        return math.sqrt((n.x - cx)**2 + (n.y - cy)**2)
 
    order_ids = list({rn.order_id for rn in route})
    scored    = sorted(order_ids, key=dist_to_center)
    to_remove = scored[:n_remove]
    new_route = [rn for rn in route if rn.order_id not in to_remove]
    return new_route, to_remove
 
 
# -----------------------------------------------------------------------------
# Repair operators
# -----------------------------------------------------------------------------
 
# def repair_greedy(route, removed_ids, orders_map, vehicle, rng):
#     """
#     Greedy insertion: insert each removed order at its cheapest position.
#     Insertion order is randomised to avoid bias.
#     Uses fast cost estimate — no check_route calls.
#     """
#     for oid in rng.sample(removed_ids, len(removed_ids)):
#         order = orders_map[oid]
#         _, route = _best_insertion(route, order, orders_map)
#     return route
# def repair_greedy(route, removed_ids, orders_map, vehicle, rng):
#     ...
#     for oid in rng.sample(removed_ids, len(removed_ids)):
#         order = orders_map[oid]
#         _, route = _best_insertion(route, order, orders_map, vehicle=vehicle)
#     return route
def repair_greedy(route, removed_ids, orders_map, vehicle, rng):
    for oid in rng.sample(removed_ids, len(removed_ids)):
        order = orders_map[oid]
        _, route = _best_insertion(route, order, orders_map, max_pos=MAX_POS, vehicle=vehicle)
    return route
 
 
# def repair_criticality_based(route, removed_ids, orders_map, vehicle, rng):
#     """
#     Insert orders with tightest time windows first (most critical first).
#     Uses fast cost estimate — no check_route calls.
#     """
#     def criticality(oid):
#         o = orders_map[oid]
#         return o.pickup_node.due_time - o.pickup_node.ready_time  # smaller = more critical
 
#     ordered = sorted(removed_ids, key=criticality)
#     for oid in ordered:
#         order = orders_map[oid]
#         # _, route = _best_insertion(route, order, orders_map)
#         _, route = _best_insertion(route, order, orders_map, vehicle=vehicle)
#     return route
 
def repair_criticality_based(route, removed_ids, orders_map, vehicle, rng):
    def criticality(oid):
        o = orders_map[oid]
        return o.pickup_node.due_time - o.pickup_node.ready_time

    ordered = sorted(removed_ids, key=criticality)
    for oid in ordered:
        order = orders_map[oid]
        _, route = _best_insertion(route, order, orders_map, max_pos=MAX_POS, vehicle=vehicle)
    return route
 
def repair_regret(route, removed_ids, orders_map, vehicle, rng):
    """
    Regret-2 insertion: at each step, insert the order whose cost difference
    between best and second-best position is highest (most 'urgent' to place).
 
    Complexity: O(k * MAX_POS²) per removed order where k = len(remaining).
    Previously was O(n * k * n²) — this is dramatically faster.
    """
    remaining = list(removed_ids)
 
    while remaining:
        regrets = []
        for oid in remaining:
            order  = orders_map[oid]
            n      = len(route)
            # limit  = min(n + 1, MAX_POS)
            # limit  = n + 1
            limit = min(n + 1, MAX_POS)
            costs  = []
 
            for i in range(limit):
                for j in range(i + 1, min(limit + 1, n + 2)):
                    # c = _insertion_cost_fast(route, order, i, j, orders_map)
                    c = _insertion_cost_fast(route, order, i, j, orders_map, vehicle)
                    costs.append(c)
 
            # costs.sort()
            # # Regret = difference between best and 2nd best position
            # regret = (costs[1] - costs[0]) if len(costs) >= 2 else 0.0
            # regrets.append((regret, oid))
            costs.sort()
            # Regret = difference between best and 2nd best position
            # Guard against inf - inf = NaN (happens when all positions violate weight)
            if len(costs) >= 2 and costs[0] < float('inf'):
                regret = costs[1] - costs[0]
            else:
                regret = 0.0
            regrets.append((regret, oid))
 
        # Insert the order with highest regret first
        regrets.sort(reverse=True)
        best_oid = regrets[0][1]
 
        order = orders_map[best_oid]
        # _, route = _best_insertion(route, order, orders_map)
        # _, route = _best_insertion(route, order, orders_map, vehicle=vehicle)
        _, route = _best_insertion(route, order, orders_map, max_pos=MAX_POS, vehicle=vehicle)
        remaining.remove(best_oid)
 
    return route
 
 
# -----------------------------------------------------------------------------
# Operator registries (used by alns_env.py)
# -----------------------------------------------------------------------------
 
DESTROY_OPS = [
    destroy_random,
    destroy_worst_cost,
    destroy_shaw,
    destroy_string,
    destroy_route_segment,
]
 
REPAIR_OPS = [
    repair_greedy,
    repair_criticality_based,
    repair_regret,
]
 
N_DESTROY = len(DESTROY_OPS)
N_REPAIR  = len(REPAIR_OPS)