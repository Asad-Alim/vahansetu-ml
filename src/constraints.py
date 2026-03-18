"""
constraints.py
==============
Route feasibility checking, ETA/ETD computation, and route metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from data_loader import Node, Order, VehicleConfig, Instance, get_travel_time, get_distance


# ─────────────────────────────────────────────────────────────────────────────
# Route node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteNode:
    order_id: str
    node:     Node

    @property
    def is_pickup(self):
        return self.node.is_pickup

    @property
    def service_time(self):
        return self.node.service_time

    def __repr__(self):
        return f"{'P' if self.is_pickup else 'D'}[{self.order_id}]"


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    feasible:            bool
    weight_ok:           bool
    volume_ok:           bool
    precedence_ok:       bool
    deadline_violations: int
    total_lateness:      float        # minutes
    arrival_times:       List[float]  # ETA per stop, minutes from start
    departure_times:     List[float]  # ETD per stop, minutes from start
    messages:            List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Core checker
# ─────────────────────────────────────────────────────────────────────────────

def check_route(route: List[RouteNode],
                orders_map: Dict[str, Order],
                vehicle: VehicleConfig,
                start_time: float = 0.0) -> CheckResult:
    """
    Validate a route against all hard/soft constraints.
    Computes ETA and ETD at each stop using get_travel_time().

    Args:
        route:       Ordered list of RouteNode stops.
        orders_map:  Dict of order_id → Order.
        vehicle:     VehicleConfig with capacity limits.
        start_time:  Offset in minutes (0 during training).

    Returns:
        CheckResult with all diagnostics and ETA/ETD lists.
    """
    messages       = []
    weight_ok      = True
    volume_ok      = True
    precedence_ok  = True
    violations     = 0
    total_lateness = 0.0
    arrivals       = []
    departures     = []

    curr_weight = 0.0
    curr_volume = 0.0
    curr_time   = start_time
    picked_up   = set()

    prev_node: Node = None

    for idx, rn in enumerate(route):
        order = orders_map[rn.order_id]

        # Travel
        if prev_node is not None:
            curr_time += get_travel_time(prev_node, rn.node)

        # Wait if early
        curr_time = max(curr_time, rn.node.ready_time)
        arrivals.append(round(curr_time, 2))

        # Load/unload
        if rn.is_pickup:
            picked_up.add(rn.order_id)
            curr_weight += order.weight_kg
            curr_volume += order.volume_m3
        else:
            if rn.order_id not in picked_up:
                precedence_ok = False
                messages.append(f"PRECEDENCE: {rn.order_id} delivered before pickup (stop {idx+1})")
            curr_weight -= order.weight_kg
            curr_volume -= order.volume_m3

        # Capacity
        if curr_weight > vehicle.max_weight + 1e-6:
            weight_ok = False
            messages.append(f"WEIGHT: {curr_weight:.1f} > {vehicle.max_weight} at stop {idx+1}")
        if curr_volume > vehicle.max_volume + 1e-6:
            volume_ok = False
            messages.append(f"VOLUME: {curr_volume:.2f} > {vehicle.max_volume} at stop {idx+1}")

        # Deadline
        deadline = rn.node.due_time
        if curr_time > deadline + 1e-6:
            tardiness = curr_time - deadline
            violations += 1
            total_lateness += tardiness
            messages.append(f"LATE: {rn} +{tardiness:.1f} min at stop {idx+1}")

        # Service time
        curr_time += rn.service_time
        departures.append(round(curr_time, 2))

        prev_node = rn.node

    return CheckResult(
        feasible=weight_ok and volume_ok and precedence_ok,
        weight_ok=weight_ok,
        volume_ok=volume_ok,
        precedence_ok=precedence_ok,
        deadline_violations=violations,
        total_lateness=round(total_lateness, 2),
        arrival_times=arrivals,
        departure_times=departures,
        messages=messages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(route: List[RouteNode],
                    vehicle: VehicleConfig) -> Dict:
    """
    Compute total distance, travel time, carbon emission, and fuel cost.
    """
    total_dist = 0.0
    total_time = 0.0

    for i in range(1, len(route)):
        a = route[i - 1].node
        b = route[i].node
        total_dist += get_distance(a, b)
        total_time += get_travel_time(a, b)

    carbon    = total_dist * vehicle.emission_factor
    fuel_cost = total_dist * vehicle.fuel_cost_per_km

    return {
        "total_distance_km":     round(total_dist, 2),
        "travel_time_min":       round(total_time, 2),
        "carbon_emission_kg":    round(carbon, 3),
        "fuel_cost_inr":         round(fuel_cost, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ETA/ETD schedule formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_schedule(route: List[RouteNode],
                    arrivals: List[float],
                    departures: List[float],
                    start_hhmm: str = "08:00") -> List[Dict]:
    """
    Convert relative-minute ETA/ETD to clock time strings.

    Args:
        start_hhmm: Route start time e.g. "08:00"

    Returns:
        List of {node_id, type, district, eta, etd, service_time_min}
    """
    from datetime import datetime, timedelta
    base = datetime.strptime(start_hhmm, "%H:%M")

    schedule = []
    for i, rn in enumerate(route):
        eta_dt = base + timedelta(minutes=arrivals[i])
        etd_dt = base + timedelta(minutes=departures[i])
        schedule.append({
            "node_id":          rn.node.node_id,
            "order_id":         rn.order_id,
            "type":             "pickup" if rn.is_pickup else "delivery",
            "district":         rn.node.district,
            "eta":              eta_dt.strftime("%H:%M"),
            "etd":              etd_dt.strftime("%H:%M"),
            "service_time_min": round(departures[i] - arrivals[i], 1),
        })
    return schedule
