


"""
data_loader.py
==============
Loads synthetic PDVRPTW JSON instances produced by generate_dataset.py.
Augments each instance per episode with randomised vehicle/service/travel attributes.
 
Key design:
  - Instance JSON is loaded once at startup
  - augment_instance() is called every env.reset() to re-randomise
  - EuclideanProvider used for training (synthetic speed sampling)
  - APIProvider stub ready for production swap (one line change)
"""
 
import os, json, math, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# ROUTE_START_MIN = 480   # 08:00 — offset to convert from-midnight to from-route-start
 
 
# Vehicle catalogue: (type_name, weight_cap_kg, volume_cap_m3, fuel_weights)
#   fuel_weights = [diesel%, petrol%, electric%]
VEHICLE_CATALOGUE = [
    ("tempo",        1000,  6,  [0.35, 0.50, 0.15]),
    ("pickup_truck", 2500,  12, [0.50, 0.40, 0.10]),
    ("medium_truck", 6000,  28, [0.65, 0.25, 0.10]),
    ("large_truck",  12000, 60, [0.75, 0.20, 0.05]),
]
 
# Mileage ranges per vehicle type per fuel type  (km/litre or km/kWh)
MILEAGE_RANGES = {
    "tempo":        {"diesel": (12, 18), "petrol": (13, 19), "electric": (60, 90)},
    "pickup_truck": {"diesel": (8,  14), "petrol": (9,  14), "electric": (50, 80)},
    "medium_truck": {"diesel": (5,  10), "petrol": (6,  10), "electric": (40, 60)},
    "large_truck":  {"diesel": (3,   7), "petrol": (4,   7), "electric": (30, 50)},
}
 
# Carbon emission factors kg/km
CARBON_FACTORS = {"diesel": 0.21, "petrol": 0.18, "electric": 0.05}
 
# Fuel price ranges INR/litre (or INR/kWh for electric)
FUEL_PRICE_RANGES = {"diesel": (85, 115), "petrol": (90, 118), "electric": (7, 10)}
 
# Speed ranges km/h per route type
SPEED_RANGES = {
    "urban":   (10.0, 35.0),
    "mixed":   (25.0, 55.0),
    "highway": (45.0, 80.0),
}
# ROUTE_TYPE_WEIGHTS = [0.55, 0.30, 0.15]   # urban / mixed / highway
ROUTE_TYPE_WEIGHTS = [0.15, 0.35, 0.50]  
MIN_SPEED, MAX_SPEED = 8.0, 90.0
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class Node:
    """A single pickup or delivery stop."""
    node_id:   str
    is_pickup: bool
    x: float          # abstract coord (used for Euclidean distance)
    y: float
    lat: float        # GPS (for display and API provider)
    lon: float
    district: str
    ready_time:  float   # minutes from route start
    due_time:    float   # minutes from route start
    packaging_time:  float = 0.0   # pickup only
    loading_time:    float = 0.0   # pickup only
    unloading_time:  float = 0.0   # delivery only
 
    @property
    def service_time(self) -> float:
        if self.is_pickup:
            return self.packaging_time + self.loading_time
        return self.unloading_time
 
 
@dataclass
class Order:
    order_id:      str
    pickup_node:   Node
    delivery_node: Node
    weight_kg:     float
    volume_m3:     float
 
 
@dataclass
class VehicleConfig:
    vehicle_type:      str
    fuel_type:         str
    max_weight:        float
    max_volume:        float
    mileage:           float        # km/litre or km/kWh
    fuel_price:        float        # INR/litre or INR/kWh
    emission_factor:   float        # kg CO2 per km
    fuel_cost_per_km:  float        # = fuel_price / mileage
 
    @property
    def type_index(self) -> int:
        types = ["tempo", "pickup_truck", "medium_truck", "large_truck"]
        return types.index(self.vehicle_type) if self.vehicle_type in types else 0
 
 
@dataclass
class Instance:
    """One fully loaded problem instance."""
    instance_id: str
    orders:      List[Order]
    vehicle:     VehicleConfig
    depot:       Node
    # Precomputed matrices (indexed by node list position)
    all_nodes:   List[Node] = field(default_factory=list)  # depot first
    node_index:  Dict[str, int] = field(default_factory=dict)
    dist_matrix: List[List[float]] = field(default_factory=list)
    time_matrix: List[List[float]] = field(default_factory=list)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Distance provider abstraction
# ─────────────────────────────────────────────────────────────────────────────
 
class DistanceProvider:
    def get_distance(self, a: Node, b: Node) -> float:
        raise NotImplementedError
 
    def get_travel_time(self, a: Node, b: Node) -> float:
        raise NotImplementedError
 
    def reset_cache(self):
        pass
 
 
class EuclideanProvider(DistanceProvider):
    """
    Training provider: Euclidean distance + synthetic Indian traffic speed.
 
    reset_cache() must be called at every env.reset() for fresh speed sampling.
 
    Pass a seed to reset_cache() whenever two solvers must see identical travel
    times (benchmark comparisons, API side-by-side runs).  Without a seed the
    RNG is re-seeded from the system clock, which is the desired behaviour
    during RL training (fresh randomness each episode).
    """
    def __init__(self):
        self._rng = random.Random()
        self._cache: Dict[Tuple, float] = {}
 
    def reset_cache(self, seed=None):
        """
        Clear the travel-time cache and optionally re-seed the internal RNG.
 
        Args:
            seed: If provided, the internal RNG is re-seeded with this value,
                  making subsequent get_travel_time() calls deterministic.
                  Pass the same seed before running Greedy and PPO on the
                  same instance so both algorithms see identical travel times.
                  Leave as None during training so each episode gets fresh
                  random speeds.
        """
        self._cache.clear()
        if seed is not None:
            self._rng = random.Random(seed)
 
    # def get_distance(self, a: Node, b: Node) -> float:
    #     return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    def get_distance(self, a: Node, b: Node) -> float:
        phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
        dphi = math.radians(b.lat - a.lat)
        dlam = math.radians(b.lon - a.lon)
        h = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return 2 * 6371.0 * math.asin(math.sqrt(h))
#  for training
    # def get_travel_time(self, a: Node, b: Node) -> float:
    #     key = (a.node_id, b.node_id)
    #     if key in self._cache:
    #         return self._cache[key]
    #     dist = self.get_distance(a, b)
    #     if dist < 1e-9:
    #         return 0.0
    #     route_type = self._rng.choices(
    #         ["urban", "mixed", "highway"], weights=ROUTE_TYPE_WEIGHTS
    #     )[0]
    #     lo, hi = SPEED_RANGES[route_type]
    #     speed = self._rng.uniform(lo, hi)
    #     speed = max(MIN_SPEED, min(MAX_SPEED, speed))
    #     tt = dist / speed * 60.0   # minutes (treating coord units as km-proxy)
    #     self._cache[key] = tt
    #     return tt
    def get_travel_time(self, a: Node, b: Node) -> float:
        key = (a.node_id, b.node_id)
        if key in self._cache:
            return self._cache[key]
        dist = self.get_distance(a, b)
        if dist < 1e-9:
            return 0.0
        # Deterministic speed derived from node IDs — same pair always gets same speed.
        # Uses a hash so no seed/cache management is needed. Replace this block
        # with a real routing API call when available.
        h = hash((min(a.node_id, b.node_id), max(a.node_id, b.node_id))) & 0xFFFFFF
        frac = (h % 1000) / 1000.0          # 0.0 – 0.999, stable per pair
        route_type = ["urban", "mixed", "highway"][h % 3]
        lo, hi = SPEED_RANGES[route_type]
        speed = lo + frac * (hi - lo)
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
        tt = dist / speed * 60.0
        self._cache[key] = tt
        return tt
 
 
class APIProvider(DistanceProvider):
    """
    Production provider: swap in for EuclideanProvider with one line.
    Calls OpenRouteService / Google Maps to get real travel times.
 
    Usage:
        from data_loader import set_provider
        set_provider(APIProvider(api_key="YOUR_KEY"))
    """
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
 
    def get_distance(self, a: Node, b: Node) -> float:
        # TODO: call routing API for road distance in km
        # e.g. OpenRouteService: GET /v2/matrix/driving-car
        raise NotImplementedError("Implement API distance call here")
 
    def get_travel_time(self, a: Node, b: Node) -> float:
        # TODO: call routing API for travel time in minutes
        # e.g. response["durations"][0][1] / 60.0
        raise NotImplementedError("Implement API travel time call here")
 
 
# ── Global provider (swap with set_provider() for production) ──────────────
_provider = EuclideanProvider()
 
def set_provider(p: DistanceProvider):
    global _provider
    _provider = p
 
def get_distance(a: Node, b: Node) -> float:
    return _provider.get_distance(a, b)
 
def get_travel_time(a: Node, b: Node) -> float:
    return _provider.get_travel_time(a, b)
 
def reset_travel_cache(seed=None):
    """
    Clear the travel-time cache.  Pass a seed to also pin the provider's RNG
    so the next call sequence is fully deterministic (used by benchmark and API
    when comparing solvers on equal footing).
    """
    _provider.reset_cache(seed=seed)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Service time sampler (triangular, mode 30–60 min as confirmed)
# ─────────────────────────────────────────────────────────────────────────────
 
# def _sample_service_times(is_pickup: bool, weight_kg: float,
#                            rng: random.Random) -> Dict:
#     weight_factor = min(2.0, 1.0 + weight_kg / 100.0)
#     if is_pickup:
#         mode_p    = rng.uniform(30.0, 60.0)
#         mode_l    = rng.uniform(30.0, 60.0)
#         packaging = round(min(120.0, max(5.0, rng.triangular(5, 120, mode_p) * weight_factor)), 1)
#         loading   = round(min(120.0, max(5.0, rng.triangular(5, 120, mode_l) * weight_factor)), 1)
#         return {"packaging_time": packaging, "loading_time": loading, "unloading_time": 0.0}
#     else:
#         mode_u    = rng.uniform(30.0, 60.0)
#         unloading = round(min(120.0, max(5.0, rng.triangular(5, 120, mode_u) * weight_factor)), 1)
#         return {"packaging_time": 0.0, "loading_time": 0.0, "unloading_time": unloading}
 
# def _sample_service_times(is_pickup: bool, weight_kg: float,
#                            rng: random.Random) -> Dict:
#     weight_factor = min(1.3, 1.0 + weight_kg / 200.0)  # was /100, much smaller now
#     if is_pickup:
#         packaging = round(min(20.0, max(3.0, rng.triangular(3, 20, 8) * weight_factor)), 1)
#         loading   = round(min(25.0, max(5.0, rng.triangular(5, 25, 12) * weight_factor)), 1)
#         return {"packaging_time": packaging, "loading_time": loading, "unloading_time": 0.0}
#     else:
#         unloading = round(min(30.0, max(5.0, rng.triangular(5, 30, 15) * weight_factor)), 1)
#         return {"packaging_time": 0.0, "loading_time": 0.0, "unloading_time": unloading}
 
#     # After re-sampling service times, clamp to safe maximums
#     for order in instance.orders:
#         order.pickup_node.packaging_time  = min(order.pickup_node.packaging_time,  15.0)
#         order.pickup_node.loading_time    = min(order.pickup_node.loading_time,    20.0)
#         order.delivery_node.unloading_time = min(order.delivery_node.unloading_time, 25.0)
 
# ─────────────────────────────────────────────────────────────────────────────
# Instance loader
# ─────────────────────────────────────────────────────────────────────────────
 
def load_instance(json_path: str) -> Optional[Instance]:
    """Load a synthetic JSON instance into an Instance object."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Cannot load {json_path}: {e}")
        return None
 
    # ── Depot ──
    # depot_data = next(n for n in data["nodes"] if n["node_type"] == "depot")
    # depot = Node(
    #     node_id="depot", is_pickup=False,
    #     x=depot_data["lon"],  y=depot_data["lat"],
    #     lat=depot_data["lat"], lon=depot_data["lon"],
    #     district=depot_data["district"],
    #     ready_time=0.0, due_time=600.0,
    # )
 
    # Replace the broken depot loader with:
    depot_loc = data["vehicle"]["current_location"]
    depot = Node(
        node_id="depot", is_pickup=False,
        x=depot_loc["lon"], y=depot_loc["lat"],
        lat=depot_loc["lat"], lon=depot_loc["lon"],
        district=depot_loc["district"],
        ready_time=0.0, due_time=1440.0,  # full day
    )
 
    # ── Vehicle (base, will be re-augmented each episode) ──
    v = data["vehicle"]
    vehicle = VehicleConfig(
        vehicle_type=v["vehicle_type"],
        fuel_type=v["fuel_type"],
        max_weight=v["capacity_weight_kg"],
        max_volume=v["capacity_volume_m3"],
        mileage=v["mileage"],
        fuel_price=v["fuel_price_per_unit"],
        emission_factor=v["carbon_per_km"],
        fuel_cost_per_km=round(v["fuel_price_per_unit"] / v["mileage"], 4),
    )
 
    # ── Orders ──
    orders = []
    for o in data["orders"]:
        p_raw = o["pickup_node"]
        d_raw = o["delivery_node"]
        pw = o["time_window"] if "time_window" in o else {}
 
        pickup = Node(
            node_id=p_raw["node_id"], is_pickup=True,
            x=p_raw["lon"], y=p_raw["lat"],
            lat=p_raw["lat"], lon=p_raw["lon"],
            district=p_raw["district"],
            ready_time=p_raw["time_window"]["open_min"],
            due_time=p_raw["time_window"]["close_min"],
            packaging_time=p_raw.get("packaging_time_min", 30.0),
            loading_time=p_raw.get("loading_time_min", 30.0),
        )
        delivery = Node(
            node_id=d_raw["node_id"], is_pickup=False,
            x=d_raw["lon"], y=d_raw["lat"],
            lat=d_raw["lat"], lon=d_raw["lon"],
            district=d_raw["district"],
            ready_time=d_raw["time_window"]["open_min"],
            due_time=d_raw["time_window"]["close_min"],
            unloading_time=d_raw.get("unloading_time_min", 30.0),
        )
        orders.append(Order(
            order_id=o["order_id"],
            pickup_node=pickup,
            delivery_node=delivery,
            weight_kg=o["weight_kg"],
            volume_m3=o["volume_m3"],
        ))
 
    # ── Build flat node list (depot first, then pickup/delivery alternating) ──
    all_nodes = [depot]
    for o in orders:
        all_nodes.append(o.pickup_node)
        all_nodes.append(o.delivery_node)
    node_index = {n.node_id: i for i, n in enumerate(all_nodes)}
 
    return Instance(
        instance_id=data["instance_id"],
        orders=orders,
        vehicle=vehicle,
        depot=depot,
        all_nodes=all_nodes,
        node_index=node_index,
        dist_matrix=[],
        time_matrix=[],
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Augmentation — called every env.reset()
# ─────────────────────────────────────────────────────────────────────────────
 
# def augment_instance(instance: Instance, rng: random.Random) -> Instance:
#     """
#     Re-randomise vehicle attributes and service times each episode.
#     Returns the SAME instance object (mutated in-place) for efficiency.
#     """
#     # ── Sample vehicle type ──
#     vtype, wt_cap, vol_cap, fuel_wts = random.choice(VEHICLE_CATALOGUE)
 
def augment_instance(instance: Instance, rng: random.Random) -> Instance:
    """
    Re-randomise vehicle attributes and service times each episode.
    Returns the SAME instance object (mutated in-place) for efficiency.
    """
    # ── Sample vehicle type (only vehicles that can carry the full load) ──
    total_weight = sum(o.weight_kg for o in instance.orders)
    total_volume = sum(o.volume_m3 for o in instance.orders)
    eligible = [
        (vtype, wt_cap, vol_cap, fuel_wts)
        for vtype, wt_cap, vol_cap, fuel_wts in VEHICLE_CATALOGUE
        if wt_cap >= total_weight and vol_cap >= total_volume
    ]
    if not eligible:
        eligible = VEHICLE_CATALOGUE  # fallback
    vtype, wt_cap, vol_cap, fuel_wts = rng.choice(eligible)
    fuel_type = rng.choices(["diesel", "petrol", "electric"], weights=fuel_wts)[0]
    lo, hi = MILEAGE_RANGES[vtype][fuel_type]
    mileage = round(rng.uniform(lo, hi), 2)
    fp_lo, fp_hi = FUEL_PRICE_RANGES[fuel_type]
    fuel_price = round(rng.uniform(fp_lo, fp_hi), 1)
    carbon = CARBON_FACTORS[fuel_type]
 
    instance.vehicle = VehicleConfig(
        vehicle_type=vtype,
        fuel_type=fuel_type,
        max_weight=wt_cap,
        max_volume=vol_cap,
        mileage=mileage,
        fuel_price=fuel_price,
        emission_factor=carbon,
        fuel_cost_per_km=round(fuel_price / mileage, 4),
    )
 
    # ── Re-sample service times ──
    # for order in instance.orders:
    #     st_p = _sample_service_times(True,  order.weight_kg, rng)
    #     order.pickup_node.packaging_time = st_p["packaging_time"]
    #     order.pickup_node.loading_time   = st_p["loading_time"]
 
    #     st_d = _sample_service_times(False, order.weight_kg, rng)
    #     order.delivery_node.unloading_time = st_d["unloading_time"]
 
    # ── Reset travel time cache ──
    # NOTE: We intentionally do NOT reset the cache here any more.
    # During RL training, ALNSEnv.reset() calls reset_travel_cache() with no
    # seed, so each episode still gets fresh random speeds (old behaviour).
    # During benchmarking and API calls, the caller resets the cache with an
    # explicit seed BEFORE invoking each solver, so both Greedy and PPO see
    # identical travel times for the same instance (fair comparison).
 
    return instance
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────
 
def build_dataset(instances_dir: str, max_files: int = 50) -> List[Instance]:
    """
    Load all JSON instances from the synthetic dataset directory.
    Returns list of Instance objects ready for training.
    """
    files = sorted([
        os.path.join(instances_dir, f)
        for f in os.listdir(instances_dir)
        if f.endswith(".json")
    ])[:max_files]
 
    instances = []
    for fp in files:
        inst = load_instance(fp)
        # if inst and len(inst.orders) >= 5:
        #if inst and len(inst.orders) >= 1:
        if inst and len(inst.orders) >= 10:
            instances.append(inst)
 
    print(f"[DataLoader] Loaded {len(instances)} instances from {instances_dir}")
    return instances