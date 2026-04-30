


"""
api_server.py
=============

FastAPI microservice — VahanSetu PPO-ALNS Route Optimiser

Supports:
- Fresh optimization (no route)
- Incremental optimization (existing route + new order)

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
"""

import os, sys, datetime, random
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from data_loader import (
    Node, Order, VehicleConfig, Instance,
    reset_travel_cache,
)
from alns_env       import ALNSEnv, Solution, build_tw_sorted_route
from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
from constraints    import RouteNode, check_route, compute_metrics, format_schedule


# ─────────────────────────────────────────────────────────────
# PPO MODEL
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# PPO_MODEL_PATH = os.getenv(
#     "PPO_MODEL_PATH",
#     os.path.join(BASE_DIR, "models_v4", "ppo_alns_final")
# )
PPO_MODEL_PATH = os.getenv(
    "PPO_MODEL_PATH",
    os.path.join(BASE_DIR, "models_v5", "ppo_alns_final")
)
_ppo_model = None

def _load_ppo():
    global _ppo_model
    try:
        from stable_baselines3 import PPO as SB3PPO
        _ppo_model = SB3PPO.load(PPO_MODEL_PATH)
        print(f"[INFO] PPO model loaded")
    except Exception as e:
        print(f"[WARN] PPO not loaded: {e}")

_load_ppo()

app = FastAPI(title="VahanSetu ALNS Microservice", version="2.0.0")


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────

class NodeIn(BaseModel):
    node_id: str
    is_pickup: bool
    x: float
    y: float
    lat: float
    lon: float
    district: str
    ready_time: float
    due_time: float
    service_time: float = 10.0


class OrderIn(BaseModel):
    order_id: str
    pickup_node: NodeIn
    delivery_node: NodeIn
    weight_kg: float
    volume_m3: float


class VehicleIn(BaseModel):
    vehicle_type: str = "pickup_truck"
    fuel_type: str = "diesel"
    max_weight: float = 2500
    max_volume: float = 12
    mileage: float = 10
    fuel_price: float = 100
    emission_factor: float = 0.21
    fuel_cost_per_km: float = 10


class OptimizeRequest(BaseModel):
    courier_id: str
    orders: List[OrderIn]
    vehicle: VehicleIn
    start_time: str = "08:00"
    use_ppo: bool = True
    max_iter: int = 50

    courier_lat: Optional[float] = None
    courier_lon: Optional[float] = None

    # 🔥 NEW
    existing_route: Optional[List[dict]] = None


# ─────────────────────────────────────────────────────────────
# CONVERTERS
# ─────────────────────────────────────────────────────────────

def _to_node(n: NodeIn) -> Node:
    node = Node(
        node_id=n.node_id,
        is_pickup=n.is_pickup,
        x=n.x, y=n.y,
        lat=n.lat, lon=n.lon,
        district=n.district,
        ready_time=n.ready_time,
        due_time=n.due_time,
    )
    if n.is_pickup:
        node.packaging_time = n.service_time / 2
        node.loading_time   = n.service_time / 2
    else:
        node.unloading_time = n.service_time
    return node


def _to_order(o: OrderIn) -> Order:
    return Order(
        order_id=o.order_id,
        pickup_node=_to_node(o.pickup_node),
        delivery_node=_to_node(o.delivery_node),
        weight_kg=o.weight_kg,
        volume_m3=o.volume_m3,
    )


def _to_vehicle(v: VehicleIn) -> VehicleConfig:
    return VehicleConfig(
        vehicle_type=v.vehicle_type,
        fuel_type=v.fuel_type,
        max_weight=v.max_weight,
        max_volume=v.max_volume,
        mileage=v.mileage,
        fuel_price=v.fuel_price,
        emission_factor=v.emission_factor,
        fuel_cost_per_km=v.fuel_cost_per_km,
    )


# ─────────────────────────────────────────────────────────────
# BUILD INSTANCE
# ─────────────────────────────────────────────────────────────

def _build_instance(req: OptimizeRequest) -> Instance:
    orders = [_to_order(o) for o in req.orders]
    vehicle = _to_vehicle(req.vehicle)

    if req.courier_lat is not None and req.courier_lon is not None:
        lat, lon = req.courier_lat, req.courier_lon
    else:
        lat = sum(o.pickup_node.lat for o in orders) / len(orders)
        lon = sum(o.pickup_node.lon for o in orders) / len(orders)

    depot = Node(
        node_id="START",
        is_pickup=False,
        x=lat, y=lon,
        lat=lat, lon=lon,
        district="Current Location",
        ready_time=0,
        due_time=1440,
    )

    return Instance(
        instance_id=req.courier_id,
        orders=orders,
        vehicle=vehicle,
        depot=depot,
    )


# ─────────────────────────────────────────────────────────────
# EXISTING ROUTE PARSER
# ─────────────────────────────────────────────────────────────

# def _build_route_from_existing(existing_route, orders_map):
#     route = []

#     for stop in existing_route:
#         order_id = stop["order_id"]
#         node_id  = stop["node_id"]

#         order = orders_map.get(order_id)
#         if not order:
#             continue

#         if node_id == order.pickup_node.node_id:
#             node = order.pickup_node
#         elif node_id == order.delivery_node.node_id:
#             node = order.delivery_node
#         else:
#             continue

#         route.append(RouteNode(order_id=order_id, node=node))

#     return route

def _build_route_from_existing(existing_route, orders_map):
    route = []

    for stop in existing_route:

        # ✅ ADD THIS VALIDATION HERE
        if "order_id" not in stop or "node_id" not in stop:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid existing_route format: {stop}"
            )

        order_id = stop["order_id"]
        node_id  = stop["node_id"]

        order = orders_map.get(order_id)
        if not order:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown order_id in existing_route: {order_id}"
            )

        if node_id == order.pickup_node.node_id:
            node = order.pickup_node
        elif node_id == order.delivery_node.node_id:
            node = order.delivery_node
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid node_id {node_id} for order {order_id}"
            )

        route.append(RouteNode(order_id=order_id, node=node))

    return route


# ─────────────────────────────────────────────────────────────
# SOLVERS
# ─────────────────────────────────────────────────────────────

# def _run_greedy(instance, max_iter, initial_route=None):
#     rng = random.Random(42)
#     orders_map = {o.order_id: o for o in instance.orders}
def _run_greedy(instance, max_iter, initial_route=None, seed=42):
    # seed is passed in from the endpoint so greedy and PPO both
    # use operator-selection randomness from the same starting point.
    rng = random.Random(seed)
    orders_map = {o.order_id: o for o in instance.orders}

    # 🔥 EDGE CASE: empty route or no existing route
    if not initial_route:
        route = build_tw_sorted_route(instance.orders, orders_map)
    else:
        route = initial_route

    current = Solution(route, orders_map, instance.vehicle)
    best = current.copy()

    for _ in range(max_iter):
        d = rng.randint(0, N_DESTROY - 1)
        r = rng.randint(0, N_REPAIR - 1)
        # n = rng.randint(1, max(1, len(instance.orders)))
        n = rng.randint(1, min(6, max(1, len(instance.orders))))


        new_route, removed = DESTROY_OPS[d](
            current.route, orders_map, instance.vehicle, n, rng)

        new_route = REPAIR_OPS[r](
            new_route, removed, orders_map, instance.vehicle, rng)

        candidate = Solution(new_route, orders_map, instance.vehicle)

        if candidate.cost() < current.cost():
            current = candidate
            if candidate.cost() < best.cost():
                best = candidate.copy()

    return best


# def _run_ppo(instance, max_iter, initial_route=None):
#     if _ppo_model is None:
#         return _run_greedy(instance, max_iter, initial_route)

#     env = ALNSEnv([instance], max_iter=max_iter)

#     if initial_route:
#         orders_map = {o.order_id: o for o in instance.orders}
#         env.current = Solution(initial_route, orders_map, instance.vehicle)
#         env.best = env.current.copy()

#     obs, _ = env.reset()

#     for _ in range(max_iter):
#         action, _ = _ppo_model.predict(obs, deterministic=True)
#         obs, _, done, _, _ = env.step(int(action))
#         if done:
#             break

#     return env.best
# def _run_ppo(instance, max_iter, initial_route=None):
#     if _ppo_model is None:
#         return _run_greedy(instance, max_iter, initial_route)

#     env = ALNSEnv([instance], max_iter=max_iter)

#     # ❌ DO NOT call env.reset()

#     orders_map = {o.order_id: o for o in instance.orders}

#     # ✅ Initialize solution manually
#     if initial_route:
#         env.current = Solution(initial_route, orders_map, instance.vehicle)
#     else:
#         env.current = Solution(
#             build_tw_sorted_route(instance.orders, orders_map),
#             orders_map,
#             instance.vehicle
#         )

#     env.best = env.current.copy()

#     # ✅ Now generate observation from current state
#     obs = env._get_obs()   # or env.get_obs() depending on your implementation

#     # ✅ Run PPO
#     for _ in range(max_iter):
#         action, _ = _ppo_model.predict(obs, deterministic=True)
#         obs, _, done, _, _ = env.step(int(action))
#         if done:
#             break

#     return env.best


def _run_ppo(instance, max_iter, initial_route=None, seed=42):
    if _ppo_model is None:
        return _run_greedy(instance, max_iter, initial_route, seed=seed)

    import numpy as _np
    from alns_operators import N_DESTROY, N_REPAIR

    env = ALNSEnv([instance], max_iter=max_iter, seed=seed)

    orders_map = {o.order_id: o for o in instance.orders}

    if initial_route:
        env.current = Solution(initial_route, orders_map, instance.vehicle)
    else:
        env.current = Solution(
            build_tw_sorted_route(instance.orders, orders_map),
            orders_map,
            instance.vehicle
        )

    # Do NOT call reset() — it calls augment_instance() which randomly
    # replaces the vehicle. Initialize all env state manually instead.
    env.instance      = instance
    env.orders_map    = orders_map
    env.best          = env.current.copy()
    env.init_cost     = max(env.current.cost(), 1.0)
    env.iteration     = 0
    env.destroy_usage = _np.zeros(N_DESTROY)
    env.repair_usage  = _np.zeros(N_REPAIR)
    env.best_found_at = 0

    obs = env._state()  # _get_obs() does not exist — correct name is _state()

    for _ in range(max_iter):
        action, _ = _ppo_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            break

    return env.best


# def _run_ppo(instance, max_iter, initial_route=None):
#     if _ppo_model is None:
#         return _run_greedy(instance, max_iter, initial_route)

#     env = ALNSEnv([instance], max_iter=max_iter)

#     # ✅ FIRST reset environment
#     obs, _ = env.reset()

#     # ✅ THEN inject your existing route
#     if initial_route:
#         orders_map = {o.order_id: o for o in instance.orders}
#         env.current = Solution(initial_route, orders_map, instance.vehicle)
#         env.best = env.current.copy()

#     # ✅ Run PPO
#     for _ in range(max_iter):
#         action, _ = _ppo_model.predict(obs, deterministic=True)
#         obs, _, done, _, _ = env.step(int(action))
#         if done:
#             break

#     return env.best


# ─────────────────────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────────────────────
def _build_response(instance, solution, start_time):
    # Convert "HH:MM" string to float minutes
    if isinstance(start_time, str):
        h, m = start_time.split(":")
        start_time_min = float(int(h) * 60 + int(m))
    else:
        start_time_min = float(start_time)

    orders_map = {o.order_id: o for o in instance.orders}

    # result  = check_route(solution.route, orders_map, instance.vehicle, start_time_min)
    # metrics = compute_metrics(solution.route, instance.vehicle, result.arrival_times, result.departure_times)
    result  = check_route(solution.route, orders_map, instance.vehicle, 0.0)
    metrics = compute_metrics(solution.route, instance.vehicle, result.arrival_times, result.departure_times)


    sched = format_schedule(
        solution.route,
        result.arrival_times,
        result.departure_times,
        start_time if isinstance(start_time, str) else f"{int(start_time_min)//60:02d}:{int(start_time_min)%60:02d}"
    )

    for i, rn in enumerate(solution.route):
        sched[i]["lat"] = rn.node.lat
        sched[i]["lon"] = rn.node.lon

    curr_w = 0.0
    peak_w = 0.0
    for rn in solution.route:
        order = solution.orders_map[rn.order_id]
        if rn.is_pickup:
            curr_w += order.weight_kg
        else:
            curr_w -= order.weight_kg
        peak_w = max(peak_w, curr_w)

    weight_util = (peak_w / instance.vehicle.max_weight * 100) if instance.vehicle.max_weight else 0.0

    return {
        "instance_id": instance.instance_id,
        "generated_at": datetime.datetime.now().isoformat(),
        "route_start_time": start_time,
        "region": "Delhi NCR",
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
        "route": sched,
        "districts_visited": sorted(set(s["district"] for s in sched)),
        "summary": {
            "total_distance_km":      metrics["total_distance_km"],
            "travel_time_min":        metrics["travel_time_min"],
            "carbon_emission_kg":     metrics["carbon_emission_kg"],
            "fuel_cost_inr":          metrics["fuel_cost_inr"],
            "num_orders":             len(instance.orders),
            "num_stops":              len(solution.route),
            "deadline_violations":    result.deadline_violations,
            "total_lateness_min":     round(result.total_lateness, 2),
            "feasible":               result.feasible,
            "objective_cost":         round(solution.cost(), 4),
            "weight_utilisation_pct": round(weight_util, 1),
        }
    }

# def _build_response(instance, solution, start_time):
#     orders_map = {o.order_id: o for o in instance.orders}

#     result  = check_route(solution.route, orders_map, instance.vehicle)
#     metrics = compute_metrics(solution.route, instance.vehicle)

#     sched = format_schedule(
#         solution.route,
#         result.arrival_times,
#         result.departure_times,
#         start_time
#     )

#     for i, rn in enumerate(solution.route):
#         sched[i]["lat"] = rn.node.lat
#         sched[i]["lon"] = rn.node.lon

#     return {
#         "instance_id": instance.instance_id,
#         "generated_at": datetime.datetime.now().isoformat(),
#         "route_start_time": start_time,
#         "region": "Delhi NCR",
#         "route": sched,
#         "summary": {
#             "total_distance_km": metrics["total_distance_km"],
#             "travel_time_min": metrics["travel_time_min"],
#             "feasible": result.feasible,
#             "objective_cost": solution.cost()
#         }
#     }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "ppo_loaded": _ppo_model is not None}


# @app.post("/optimize")
# def optimize(req: OptimizeRequest):

#     if not req.orders:
#         raise HTTPException(400, "orders empty")

#     reset_travel_cache()
#     instance = _build_instance(req)
#     orders_map = {o.order_id: o for o in instance.orders}

#     # 🔥 HANDLE EXISTING ROUTE
#     initial_route = None
#     if req.existing_route:
#         initial_route = _build_route_from_existing(
#             req.existing_route, orders_map
#         )

#     # 🔥 RUN
#     solution = (
#         _run_ppo(instance, req.max_iter, initial_route)
#         if req.use_ppo
#         else _run_greedy(instance, req.max_iter, initial_route)
#     )

#     return _build_response(instance, solution, req.start_time)

@app.post("/optimize")
def optimize(req: OptimizeRequest):
 
    if not req.orders:
        raise HTTPException(400, "orders empty")
 
    instance = _build_instance(req)
    orders_map = {o.order_id: o for o in instance.orders}
 
    # Derive a deterministic travel-time seed from the request payload.
    # Using the number of orders + their combined weight means the same
    # logical request always produces the same A→B speeds, while different
    # requests get different (but still deterministic) speed maps.
    # Both _run_greedy and _run_ppo receive the SAME travel_seed so the
    # road conditions are identical regardless of which solver is chosen.
    travel_seed = hash(tuple(o.order_id for o in instance.orders)) & 0xFFFFFF
    reset_travel_cache(seed=travel_seed)
 
    # 🔥 HANDLE EXISTING ROUTE
    # initial_route = None
    # if req.existing_route:
    #     initial_route = _build_route_from_existing(
    #         req.existing_route, orders_map
    #     )
    # In api_server.py, in the optimize() endpoint, replace the existing_route block:

    initial_route = None
    if req.existing_route:
        initial_route = _build_route_from_existing(req.existing_route, orders_map)
        
        # Insert any new orders not already in the existing route
        route_order_ids = {rn.order_id for rn in initial_route}
        new_orders = [o for o in instance.orders if o.order_id not in route_order_ids]
        
        if new_orders:
            from alns_operators import _best_insertion
            for new_order in new_orders:
                _, initial_route = _best_insertion(
                    initial_route, new_order, orders_map, vehicle=instance.vehicle
                )
 
    # 🔥 RUN  (pass travel_seed as operator seed too for full reproducibility)
    # solution = (
    #     _run_ppo(instance, req.max_iter, initial_route)
    #     if req.use_ppo
    #     else _run_greedy(instance, req.max_iter, initial_route, seed=travel_seed)
    # )

    solution = (
        _run_ppo(instance, req.max_iter, initial_route, seed=travel_seed)
        if req.use_ppo
        else _run_greedy(instance, req.max_iter, initial_route, seed=travel_seed)
    )
 
    return _build_response(instance, solution, req.start_time)