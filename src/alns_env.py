# # """
# # alns_env.py
# # ===========
# # Gymnasium environment for PPO-guided ALNS on PDVRPTW.

# # State vector (17 features):
# #   [0]  search_progress          (iteration / max_iter)
# #   [1]  solution_delta           (current_cost - best_cost) / init_cost
# #   [2]  init_cost_norm           (init_cost / MAX_COST)
# #   [3]  best_cost_norm           (best_cost / MAX_COST)
# #   [4-8] destroy_usage×5        (usage count / iteration, normalised)
# #   [9-11] repair_usage×3        (usage count / iteration, normalised)
# #   [12]  avg_tw_norm             (avg time window width / MAX_TW)
# #   [13]  avg_service_norm        (avg service time / MAX_SERVICE)
# #   [14]  fuel_cost_per_km_norm   (fuel_cost_per_km / MAX_FUEL_COST)
# #   [15]  vehicle_type_enc        (0–3 normalised)

# # Action: [destroy_index (0–4), repair_index (0–2)]
# #   → encoded as single integer: action = d * N_REPAIR + r
# #   → total 15 discrete actions
# # """

# # import random
# # import numpy as np
# # try:
# #     import gymnasium as gym
# #     from gymnasium import spaces
# # except ImportError:
# #     import gym
# #     from gym import spaces
# # from typing import List, Dict, Optional

# # from data_loader import Instance, Order, augment_instance, get_travel_time
# # from constraints import RouteNode, CheckResult, check_route, compute_metrics
# # from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Objective weights (Eq. 3 / 4 from Wang et al. 2025)
# # # ─────────────────────────────────────────────────────────────────────────────

# # W_TRAVEL_TIME  = 1.0
# # W_LATENESS     = 10.0
# # W_CARBON       = 0.05
# # W_FUEL         = 0.1
# # W_INFEASIBLE   = 1e6

# # # Step reward params (Eq. 3): alpha=1.0 beta=1.0 gamma=2.0
# # ALPHA, BETA, GAMMA = 1.0, 1.0, 2.0

# # # Final reward params (Eq. 4): f1=5.0 f2=0.5
# # F1, F2 = 5.0, 0.5

# # # Normalisation ceilings
# # MAX_COST      = 1e5
# # MAX_TW        = 600.0    # minutes (full working day)
# # MAX_SERVICE   = 240.0    # minutes (2× max service time)
# # MAX_FUEL_COST = 50.0     # INR/km ceiling

# # # n_remove: number of orders to destroy each step
# # MIN_REMOVE = 1
# # MAX_REMOVE = 6


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Solution container
# # # ─────────────────────────────────────────────────────────────────────────────

# # class Solution:
# #     def __init__(self, route: List[RouteNode],
# #                  orders_map: Dict[str, Order],
# #                  vehicle):
# #         self.route      = route
# #         self.orders_map = orders_map
# #         self.vehicle    = vehicle
# #         self._cost      = None

# #     def cost(self) -> float:
# #         if self._cost is not None:
# #             return self._cost
# #         result = check_route(self.route, self.orders_map, self.vehicle)
# #         metrics = compute_metrics(self.route, self.vehicle)

# #         c = (W_TRAVEL_TIME * metrics["travel_time_min"]
# #            + W_LATENESS    * result.total_lateness
# #            + W_CARBON      * metrics["carbon_emission_kg"]
# #            + W_FUEL        * metrics["fuel_cost_inr"])

# #         if not result.feasible:
# #             c += W_INFEASIBLE

# #         self._cost = round(c, 4)
# #         return self._cost

# #     def invalidate(self):
# #         self._cost = None

# #     def copy(self) -> "Solution":
# #         return Solution(list(self.route), self.orders_map, self.vehicle)


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Environment
# # # ─────────────────────────────────────────────────────────────────────────────

# # class ALNSEnv(gym.Env):
# #     metadata = {"render_modes": []}

# #     def __init__(self,
# #                  instances: List[Instance],
# #                  max_iter: int = 100,
# #                  seed: int = 42):
# #         super().__init__()

# #         self.instances = instances
# #         self.max_iter  = max_iter
# #         self.rng       = random.Random(seed)

# #         # Spaces
# #         n_actions = N_DESTROY * N_REPAIR   # 5 × 3 = 15
# #         self.action_space      = spaces.Discrete(n_actions)
# #         self.observation_space = spaces.Box(
# #             low=0.0, high=1.0, shape=(16,), dtype=np.float32
# #         )

# #         # Episode state (initialised in reset)
# #         self.instance:   Optional[Instance]  = None
# #         self.orders_map: Optional[Dict]      = None
# #         self.current:    Optional[Solution]  = None
# #         self.best:       Optional[Solution]  = None
# #         self.init_cost:  float               = 1.0
# #         self.iteration:  int                 = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)

# #     # ── Reset ──────────────────────────────────────────────────────────────

# #     def reset(self, seed=None, options=None):
# #         # Pick a random instance and re-augment it
# #         inst = self.rng.choice(self.instances)
# #         self.instance = augment_instance(inst, self.rng)
# #         self.orders_map = {o.order_id: o for o in self.instance.orders}

# #         # Build initial greedy solution (sequential pickup→delivery per order)
# #         initial_route = self._build_initial_route()
# #         self.current  = Solution(initial_route, self.orders_map, self.instance.vehicle)
# #         self.best     = self.current.copy()
# #         self.init_cost = max(self.current.cost(), 1.0)

# #         self.iteration     = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)

# #         return self._state(), {}

# #     def _build_initial_route(self) -> List[RouteNode]:
# #         route = []
# #         for o in self.instance.orders:
# #             route.append(RouteNode(o.order_id, o.pickup_node))
# #             route.append(RouteNode(o.order_id, o.delivery_node))
# #         return route

# #     # ── Step ───────────────────────────────────────────────────────────────

# #     def step(self, action: int):
# #         d_idx = action // N_REPAIR
# #         r_idx = action  % N_REPAIR
# #         d_idx = min(d_idx, N_DESTROY - 1)
# #         r_idx = min(r_idx, N_REPAIR  - 1)

# #         destroy_op = DESTROY_OPS[d_idx]
# #         repair_op  = REPAIR_OPS[r_idx]

# #         n_remove = self.rng.randint(MIN_REMOVE, min(MAX_REMOVE, len(self.instance.orders)))
# #         prev_cost = self.current.cost()

# #         # Destroy
# #         new_route, removed = destroy_op(
# #             self.current.route, self.orders_map,
# #             self.instance.vehicle, n_remove, self.rng
# #         )

# #         # Repair
# #         new_route = repair_op(
# #             new_route, removed, self.orders_map,
# #             self.instance.vehicle, self.rng
# #         )

# #         candidate = Solution(new_route, self.orders_map, self.instance.vehicle)
# #         new_cost  = candidate.cost()

# #         # Accept/reject (greedy acceptance)
# #         if new_cost < prev_cost:
# #             self.current = candidate
# #             if new_cost < self.best.cost():
# #                 self.best = candidate.copy()

# #         # Update usage
# #         self.destroy_usage[d_idx] += 1
# #         self.repair_usage[r_idx]  += 1
# #         self.iteration += 1

# #         # Reward (Eq. 3)
# #         delta = prev_cost - new_cost
# #         if delta > 0:
# #             reward = ALPHA * delta / self.init_cost
# #         elif delta == 0:
# #             reward = 0.0
# #         else:
# #             reward = -BETA * abs(delta) / self.init_cost

# #         done = self.iteration >= self.max_iter

# #         # Final bonus (Eq. 4)
# #         if done:
# #             improvement = (self.init_cost - self.best.cost()) / self.init_cost
# #             reward += F1 * improvement - F2 * check_route(
# #                 self.best.route, self.orders_map, self.instance.vehicle
# #             ).deadline_violations

# #         return self._state(), float(reward), done, False, {}

# #     # ── State ──────────────────────────────────────────────────────────────

# #     def _state(self) -> np.ndarray:
# #         it  = self.iteration
# #         sp  = it / self.max_iter
# #         curr_cost = self.current.cost()
# #         best_cost = self.best.cost()
# #         delta = (curr_cost - best_cost) / max(self.init_cost, 1.0)

# #         # Operator usage rates
# #         total = max(it, 1)
# #         d_usage = self.destroy_usage / total
# #         r_usage = self.repair_usage  / total

# #         # Instance features
# #         tws = [(o.pickup_node.due_time - o.pickup_node.ready_time)
# #                for o in self.instance.orders]
# #         avg_tw = np.mean(tws) / MAX_TW

# #         svc = [o.pickup_node.service_time + o.delivery_node.service_time
# #                for o in self.instance.orders]
# #         avg_svc = np.mean(svc) / MAX_SERVICE

# #         fck_norm = self.instance.vehicle.fuel_cost_per_km / MAX_FUEL_COST
# #         vt_norm  = self.instance.vehicle.type_index / 3.0

# #         state = np.array([
# #             sp,
# #             np.clip(delta, -1.0, 1.0),
# #             np.clip(self.init_cost / MAX_COST, 0.0, 1.0),
# #             np.clip(best_cost      / MAX_COST, 0.0, 1.0),
# #             *np.clip(d_usage, 0.0, 1.0),
# #             *np.clip(r_usage, 0.0, 1.0),
# #             np.clip(avg_tw,   0.0, 1.0),
# #             np.clip(avg_svc,  0.0, 1.0),
# #             np.clip(fck_norm, 0.0, 1.0),
# #             np.clip(vt_norm,  0.0, 1.0),
# #         ], dtype=np.float32)

# #         return state


# """
# alns_env.py
# ===========
# Gymnasium environment for PPO-guided ALNS on PDVRPTW.
 
# State vector (16 features):
#   [0]  search_progress          (iteration / max_iter)
#   [1]  solution_delta           (current_cost - best_cost) / init_cost
#   [2]  init_cost_norm           (init_cost / MAX_COST)
#   [3]  best_cost_norm           (best_cost / MAX_COST)
#   [4-8] destroy_usage×5        (usage count / iteration, normalised)
#   [9-11] repair_usage×3        (usage count / iteration, normalised)
#   [12]  avg_tw_norm             (avg time window width / MAX_TW)
#   [13]  avg_service_norm        (avg service time / MAX_SERVICE)
#   [14]  fuel_cost_per_km_norm   (fuel_cost_per_km / MAX_FUEL_COST)
#   [15]  vehicle_type_enc        (0-3 normalised)
 
# Action: [destroy_index (0-4), repair_index (0-2)]
#   -> encoded as single integer: action = d * N_REPAIR + r
#   -> total 15 discrete actions
# """
 
# # # new code starts here
# # import random
# # import numpy as np
# # try:
# #     import gymnasium as gym
# #     from gymnasium import spaces
# # except ImportError:
# #     import gym
# #     from gym import spaces
# # from typing import List, Dict, Optional
 
# # from data_loader import Instance, Order, augment_instance, get_travel_time
# # from constraints import RouteNode, CheckResult, check_route, compute_metrics
# # from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
 
 
# # # -----------------------------------------------------------------------------
# # # Objective weights
# # # -----------------------------------------------------------------------------
 
# # W_TRAVEL_TIME  = 1.0
# # W_LATENESS     = 25.0
# # W_CARBON       = 0.05
# # W_FUEL         = 0.1
# # W_INFEASIBLE   = 1e6
 
# # # Step reward params (Eq. 3 — Wang et al. 2025)
# # # ALPHA: reward scale when improving current solution
# # # BETA:  penalty scale when worsening current solution
# # # GAMMA: reward scale when beating global best (strongest signal)
# # ALPHA, BETA, GAMMA = 1.0, 1.5, 2.5
 
# # # Final reward params (Eq. 4 — Wang et al. 2025)
# # # F1: scales the total improvement ratio achieved over the episode
# # # F2: efficiency bonus — rewards finishing with fewer wasted iterations
# # F1, F2 = 5.0, 1.0
 
# # # Normalisation ceilings
# # MAX_COST      = 1e5
# # MAX_TW        = 600.0    # minutes (full working day)
# # MAX_SERVICE   = 240.0    # minutes (2x max service time)
# # MAX_FUEL_COST = 50.0     # INR/km ceiling
 
# # # n_remove: number of orders to destroy each step
# # MIN_REMOVE = 1
# # MAX_REMOVE = 6
 
 
# # # -----------------------------------------------------------------------------
# # # Solution container
# # # -----------------------------------------------------------------------------
 
# # class Solution:
# #     def __init__(self, route: List[RouteNode],
# #                  orders_map: Dict[str, Order],
# #                  vehicle):
# #         self.route      = route
# #         self.orders_map = orders_map
# #         self.vehicle    = vehicle
# #         self._cost      = None
 
# #     def cost(self) -> float:
# #         if self._cost is not None:
# #             return self._cost
# #         result = check_route(self.route, self.orders_map, self.vehicle)
# #         metrics = compute_metrics(self.route, self.vehicle)
 
# #         c = (W_TRAVEL_TIME * metrics["travel_time_min"]
# #            + W_LATENESS    * result.total_lateness
# #            + W_CARBON      * metrics["carbon_emission_kg"]
# #            + W_FUEL        * metrics["fuel_cost_inr"])
 
# #         if not result.feasible:
# #             c += W_INFEASIBLE
 
# #         self._cost = round(c, 4)
# #         return self._cost
 
# #     def invalidate(self):
# #         self._cost = None
 
# #     def copy(self) -> "Solution":
# #         return Solution(list(self.route), self.orders_map, self.vehicle)
 
 
# # # -----------------------------------------------------------------------------
# # # Environment
# # # -----------------------------------------------------------------------------
 
# # class ALNSEnv(gym.Env):
# #     metadata = {"render_modes": []}
 
# #     def __init__(self,
# #                  instances: List[Instance],
# #                  max_iter: int = 100,
# #                  seed: int = 42):
# #         super().__init__()
 
# #         self.instances = instances
# #         self.max_iter  = max_iter
# #         self.rng       = random.Random(seed)
 
# #         # Spaces
# #         n_actions = N_DESTROY * N_REPAIR   # 5 x 3 = 15
# #         self.action_space      = spaces.Discrete(n_actions)
# #         self.observation_space = spaces.Box(
# #             low=0.0, high=1.0, shape=(16,), dtype=np.float32
# #         )
 
# #         # Episode state (initialised in reset)
# #         self.instance:   Optional[Instance]  = None
# #         self.orders_map: Optional[Dict]      = None
# #         self.current:    Optional[Solution]  = None
# #         self.best:       Optional[Solution]  = None
# #         self.init_cost:  float               = 1.0
# #         self.iteration:  int                 = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)
 
# #     # ── Reset ──────────────────────────────────────────────────────────────
 
# #     def reset(self, seed=None, options=None):
# #         inst = self.rng.choice(self.instances)
# #         self.instance = augment_instance(inst, self.rng)
# #         self.orders_map = {o.order_id: o for o in self.instance.orders}
 
# #         initial_route = self._build_initial_route()
# #         self.current  = Solution(initial_route, self.orders_map, self.instance.vehicle)
# #         self.best     = self.current.copy()
# #         self.init_cost = max(self.current.cost(), 1.0)
 
# #         self.iteration     = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)
 
# #         return self._state(), {}
 
# #     def _build_initial_route(self) -> List[RouteNode]:
# #         route = []
# #         for o in self.instance.orders:
# #             route.append(RouteNode(o.order_id, o.pickup_node))
# #             route.append(RouteNode(o.order_id, o.delivery_node))
# #         return route
 
# #     # ── Step ───────────────────────────────────────────────────────────────
 
# #     def step(self, action: int):
# #         d_idx = action // N_REPAIR
# #         r_idx = action  % N_REPAIR
# #         d_idx = min(d_idx, N_DESTROY - 1)
# #         r_idx = min(r_idx, N_REPAIR  - 1)
 
# #         destroy_op = DESTROY_OPS[d_idx]
# #         repair_op  = REPAIR_OPS[r_idx]
 
# #         n_remove = self.rng.randint(MIN_REMOVE, min(MAX_REMOVE, len(self.instance.orders)))
# #         prev_cost = self.current.cost()
# #         best_cost_before = self.best.cost()
 
# #         # Destroy
# #         new_route, removed = destroy_op(
# #             self.current.route, self.orders_map,
# #             self.instance.vehicle, n_remove, self.rng
# #         )
 
# #         # Repair
# #         new_route = repair_op(
# #             new_route, removed, self.orders_map,
# #             self.instance.vehicle, self.rng
# #         )
 
# #         candidate = Solution(new_route, self.orders_map, self.instance.vehicle)
# #         new_cost  = candidate.cost()
 
# #         # Accept/reject (greedy acceptance)
# #         if new_cost < prev_cost:
# #             self.current = candidate
# #             if new_cost < self.best.cost():
# #                 self.best = candidate.copy()
 
# #         # Update usage
# #         self.destroy_usage[d_idx] += 1
# #         self.repair_usage[r_idx]  += 1
# #         self.iteration += 1
 
# #         # ── Reward (Eq. 3 — Wang et al. 2025) ──────────────────────────────
# #         # Three cases in priority order:
# #         # Case 1: new solution beats global best → GAMMA reward (strongest)
# #         # Case 2: new solution improves current but not best → ALPHA reward
# #         # Case 3: new solution worsens current → BETA penalty
# #         delta = prev_cost - new_cost
# #         if new_cost < best_cost_before:
# #             # Beat the global best — most important event, highest reward
# #             reward = GAMMA * (best_cost_before - new_cost) / best_cost_before
# #         elif delta > 0:
# #             # Improved current solution but did not beat global best
# #             reward = ALPHA * delta / self.init_cost
# #         elif delta == 0:
# #             reward = 0.0
# #         else:
# #             # Worsened current solution
# #             reward = -BETA * abs(delta) / self.init_cost
 
# #         done = self.iteration >= self.max_iter
 
# #         # ── Final bonus (Eq. 4 — Wang et al. 2025) ─────────────────────────
# #         # F1 * improvement_ratio: rewards total cost reduction over episode
# #         # F2 * efficiency: rewards finding good solution without wasting iterations
# #         if done:
# #             improvement = (self.init_cost - self.best.cost()) / self.init_cost
# #             efficiency  =1- (self.iteration / self.max_iter)
# #             reward += F1 * improvement + F2 * efficiency
 
# #         return self._state(), float(reward), done, False, {}
 
# #     # ── State ──────────────────────────────────────────────────────────────
 
# #     def _state(self) -> np.ndarray:
# #         it  = self.iteration
# #         sp  = it / self.max_iter
# #         curr_cost = self.current.cost()
# #         best_cost = self.best.cost()
# #         delta = (curr_cost - best_cost) / max(self.init_cost, 1.0)
 
# #         total = max(it, 1)
# #         d_usage = self.destroy_usage / total
# #         r_usage = self.repair_usage  / total
 
# #         tws = [(o.pickup_node.due_time - o.pickup_node.ready_time)
# #                for o in self.instance.orders]
# #         avg_tw = np.mean(tws) / MAX_TW
 
# #         svc = [o.pickup_node.service_time + o.delivery_node.service_time
# #                for o in self.instance.orders]
# #         avg_svc = np.mean(svc) / MAX_SERVICE
 
# #         fck_norm = self.instance.vehicle.fuel_cost_per_km / MAX_FUEL_COST
# #         vt_norm  = self.instance.vehicle.type_index / 3.0
 
# #         state = np.array([
# #             sp,
# #             np.clip(delta, -1.0, 1.0),
# #             np.clip(self.init_cost / MAX_COST, 0.0, 1.0),
# #             np.clip(best_cost      / MAX_COST, 0.0, 1.0),
# #             *np.clip(d_usage, 0.0, 1.0),
# #             *np.clip(r_usage, 0.0, 1.0),
# #             np.clip(avg_tw,   0.0, 1.0),
# #             np.clip(avg_svc,  0.0, 1.0),
# #             np.clip(fck_norm, 0.0, 1.0),
# #             np.clip(vt_norm,  0.0, 1.0),
# #         ], dtype=np.float32)
 
# #         return state
 

# """
# alns_env.py
# ===========
# Gymnasium environment for PPO-guided ALNS on PDVRPTW.
 
# State vector (17 features):
#   [0]  search_progress          (iteration / max_iter)
#   [1]  solution_delta           (current_cost - best_cost) / init_cost
#   [2]  init_cost_norm           (init_cost / MAX_COST)
#   [3]  best_cost_norm           (best_cost / MAX_COST)
#   [4-8] destroy_usage x5       (usage count / iteration, normalised)
#   [9-11] repair_usage x3       (usage count / iteration, normalised)
#   [12]  avg_tw_norm             (avg time window width / MAX_TW)
#   [13]  avg_service_norm        (avg service time / MAX_SERVICE)
#   [14]  fuel_cost_per_km_norm   (fuel_cost_per_km / MAX_FUEL_COST)
#   [15]  vehicle_type_enc        (0-3 normalised)
#   [16]  violation_rate          (deadline_violations / n_orders)
 
# Action: [destroy_index (0-4), repair_index (0-2)]
#   -> encoded as single integer: action = d * N_REPAIR + r
#   -> total 15 discrete actions
 
# Reward design (Wang et al. 2025, Eq. 3 / 4):
#   Step reward:
#     Case 1  new_cost < best_ever          -> +GAMMA * relative_improvement (global best signal)
#     Case 2  prev_cost > new_cost >= best  -> +ALPHA * delta / init_cost     (local improvement)
#     Case 3  delta == 0                    ->  0
#     Case 4  new_cost > prev_cost          -> -BETA  * |delta| / init_cost   (worsening penalty)
#   Exploration bonus (training only):
#     +ETA * operator_diversity                                                (encourages trying all ops)
#   Terminal reward (Eq. 4):
#     +F1 * (init_cost - best_cost) / init_cost   (quality: how much did we improve?)
#     +F2 * (1 - iteration / max_iter)            (efficiency: did we find it early?)
# """
 

# # #  another code iteration starts here with violation_rate added to state and step reward redesigned to prioritize global best improvements, with an exploration bonus for operator diversity during training.
# # import random
# # import numpy as np
# # try:
# #     import gymnasium as gym
# #     from gymnasium import spaces
# # except ImportError:
# #     import gym
# #     from gym import spaces
# # from typing import List, Dict, Optional
 
# # from data_loader import Instance, Order, augment_instance, get_travel_time
# # from constraints import RouteNode, CheckResult, check_route, compute_metrics
# # from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
 
 
# # # -----------------------------------------------------------------------------
# # # Objective weights
# # # -----------------------------------------------------------------------------
 
# # W_TRAVEL_TIME  = 1.0
# # W_LATENESS     = 25.0       # strong penalty — violations were 105-140 in previous runs
# # W_CARBON       = 0.05
# # W_FUEL         = 0.1
# # W_INFEASIBLE   = 1e6
 
# # # -----------------------------------------------------------------------------
# # # Step reward params  (Eq. 3 — Wang et al. 2025)
# # # -----------------------------------------------------------------------------
# # # ALPHA : reward scale for improving current solution (but not global best)
# # # BETA  : penalty scale for worsening current solution
# # # GAMMA : reward scale for beating global best — highest signal, must be > ALPHA
# # # ETA   : exploration bonus scale (diversity of operator usage during training)
# # ALPHA = 1.0
# # BETA  = 1.0     # paper value; kept symmetric with ALPHA so penalties don't dominate
# # GAMMA = 2.5     # significantly higher than ALPHA so PPO strongly prefers global-best moves
# # ETA   = 0.3     # small exploration bonus — fades naturally as usage counts equalise
 
# # # -----------------------------------------------------------------------------
# # # Terminal reward params  (Eq. 4 — Wang et al. 2025)
# # # -----------------------------------------------------------------------------
# # # F1 : weight on total improvement ratio  (primary goal — find the best solution)
# # # F2 : weight on efficiency bonus         (secondary goal — find it in fewer iterations)
# # #      efficiency = 1 - t/Tmax  so earlier termination (if added) gets higher bonus
# # #      with fixed-length episodes this is a constant 0, but it becomes active
# # #      if early-stopping is enabled in the future
# # F1 = 5.0
# # F2 = 1.0
 
# # # -----------------------------------------------------------------------------
# # # Normalisation ceilings
# # # -----------------------------------------------------------------------------
# # MAX_COST      = 1e5
# # MAX_TW        = 600.0    # minutes (full working day)
# # MAX_SERVICE   = 240.0    # minutes (2x max service time)
# # MAX_FUEL_COST = 50.0     # INR/km ceiling
 
# # # n_remove: number of orders to destroy each step
# # MIN_REMOVE = 1
# # MAX_REMOVE = 6
 
 
# # # -----------------------------------------------------------------------------
# # # Solution container
# # # -----------------------------------------------------------------------------
 
# # class Solution:
# #     def __init__(self, route: List[RouteNode],
# #                  orders_map: Dict[str, Order],
# #                  vehicle):
# #         self.route      = route
# #         self.orders_map = orders_map
# #         self.vehicle    = vehicle
# #         self._cost      = None
# #         self._violations = None
 
# #     def cost(self) -> float:
# #         if self._cost is not None:
# #             return self._cost
# #         self._compute()
# #         return self._cost
 
# #     def violations(self) -> int:
# #         if self._violations is not None:
# #             return self._violations
# #         self._compute()
# #         return self._violations
 
# #     def _compute(self):
# #         result  = check_route(self.route, self.orders_map, self.vehicle)
# #         metrics = compute_metrics(self.route, self.vehicle)
 
# #         c = (W_TRAVEL_TIME * metrics["travel_time_min"]
# #            + W_LATENESS    * result.total_lateness
# #            + W_CARBON      * metrics["carbon_emission_kg"]
# #            + W_FUEL        * metrics["fuel_cost_inr"])
 
# #         if not result.feasible:
# #             c += W_INFEASIBLE
 
# #         self._cost       = round(c, 4)
# #         self._violations = getattr(result, "deadline_violations", 0)
 
# #     def invalidate(self):
# #         self._cost       = None
# #         self._violations = None
 
# #     def copy(self) -> "Solution":
# #         return Solution(list(self.route), self.orders_map, self.vehicle)
 
 
# # # -----------------------------------------------------------------------------
# # # Environment
# # # -----------------------------------------------------------------------------
 
# # class ALNSEnv(gym.Env):
# #     metadata = {"render_modes": []}
 
# #     def __init__(self,
# #                  instances: List[Instance],
# #                  max_iter: int = 100,
# #                  seed: int = 42):
# #         super().__init__()
 
# #         self.instances = instances
# #         self.max_iter  = max_iter
# #         self.rng       = random.Random(seed)
 
# #         # Spaces — 17 features (16 original + violation_rate)
# #         n_actions = N_DESTROY * N_REPAIR   # 5 x 3 = 15
# #         self.action_space      = spaces.Discrete(n_actions)
# #         self.observation_space = spaces.Box(
# #             low=0.0, high=1.0, shape=(17,), dtype=np.float32
# #         )
 
# #         # Episode state (initialised in reset)
# #         self.instance:    Optional[Instance] = None
# #         self.orders_map:  Optional[Dict]     = None
# #         self.current:     Optional[Solution] = None
# #         self.best:        Optional[Solution] = None
# #         self.init_cost:   float              = 1.0
# #         self.iteration:   int                = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)
 
# #     # ── Reset ──────────────────────────────────────────────────────────────
 
# #     def reset(self, seed=None, options=None):
# #         inst = self.rng.choice(self.instances)
# #         self.instance   = augment_instance(inst, self.rng)
# #         self.orders_map = {o.order_id: o for o in self.instance.orders}
 
# #         initial_route  = self._build_initial_route()
# #         self.current   = Solution(initial_route, self.orders_map, self.instance.vehicle)
# #         self.best      = self.current.copy()
# #         self.init_cost = max(self.current.cost(), 1.0)
 
# #         self.iteration     = 0
# #         self.destroy_usage = np.zeros(N_DESTROY)
# #         self.repair_usage  = np.zeros(N_REPAIR)
 
# #         return self._state(), {}
 
# #     def _build_initial_route(self) -> List[RouteNode]:
# #         route = []
# #         for o in self.instance.orders:
# #             route.append(RouteNode(o.order_id, o.pickup_node))
# #             route.append(RouteNode(o.order_id, o.delivery_node))
# #         return route
 
# #     # ── Step ───────────────────────────────────────────────────────────────
 
# #     def step(self, action: int):
# #         d_idx = min(action // N_REPAIR, N_DESTROY - 1)
# #         r_idx = min(action  % N_REPAIR, N_REPAIR  - 1)
 
# #         destroy_op = DESTROY_OPS[d_idx]
# #         repair_op  = REPAIR_OPS[r_idx]
 
# #         n_remove  = self.rng.randint(MIN_REMOVE, min(MAX_REMOVE, len(self.instance.orders)))
# #         prev_cost        = self.current.cost()
# #         best_cost_before = self.best.cost()
 
# #         # ── Destroy ────────────────────────────────────────────────────────
# #         new_route, removed = destroy_op(
# #             self.current.route, self.orders_map,
# #             self.instance.vehicle, n_remove, self.rng
# #         )
 
# #         # ── Repair ─────────────────────────────────────────────────────────
# #         new_route = repair_op(
# #             new_route, removed, self.orders_map,
# #             self.instance.vehicle, self.rng
# #         )
 
# #         candidate = Solution(new_route, self.orders_map, self.instance.vehicle)
# #         new_cost  = candidate.cost()
 
# #         # ── Accept / reject  (greedy acceptance) ───────────────────────────
# #         if new_cost < prev_cost:
# #             self.current = candidate
# #             if new_cost < self.best.cost():
# #                 self.best = candidate.copy()
 
# #         # ── Update usage counters ───────────────────────────────────────────
# #         self.destroy_usage[d_idx] += 1
# #         self.repair_usage[r_idx]  += 1
# #         self.iteration += 1
 
# #         # ── Step reward  (Eq. 3 — Wang et al. 2025) ────────────────────────
# #         #
# #         # Priority order matters:
# #         #   1. Global best beat  -> GAMMA (strongest — this is the primary goal)
# #         #   2. Local improvement -> ALPHA
# #         #   3. No change         -> 0
# #         #   4. Worsened          -> -BETA
# #         #
# #         delta = prev_cost - new_cost
 
# #         if new_cost < best_cost_before:
# #             # Beat the global best — most important event
# #             reward = GAMMA * (best_cost_before - new_cost) / max(best_cost_before, 1.0)
# #         elif delta > 0:
# #             # Improved current but did not beat global best
# #             reward = ALPHA * delta / self.init_cost
# #         elif delta == 0:
# #             reward = 0.0
# #         else:
# #             # Worsened current solution
# #             reward = -BETA * abs(delta) / self.init_cost
 
# #         # ── Exploration bonus ───────────────────────────────────────────────
# #         #
# #         # Encourages PPO to try all destroy/repair combinations during training
# #         # rather than collapsing onto a single favourite operator pair.
# #         #
# #         # Computed as normalised Shannon entropy of operator usage distribution.
# #         # When all operators used equally  -> entropy = max -> bonus = ETA
# #         # When only one operator used      -> entropy = 0   -> bonus = 0
# #         # Naturally fades as PPO converges to a policy (usage counts specialise).
# #         #
# #         all_usage = np.concatenate([self.destroy_usage, self.repair_usage])
# #         total_use = all_usage.sum()
# #         if total_use > 0:
# #             probs       = all_usage / total_use
# #             probs       = probs[probs > 0]
# #             entropy     = -np.sum(probs * np.log(probs))
# #             max_entropy = np.log(N_DESTROY + N_REPAIR)   # entropy under uniform distribution
# #             diversity   = entropy / max_entropy            # 0 = concentrated, 1 = uniform
# #             reward     += ETA * diversity
 
# #         # ── Terminal reward  (Eq. 4 — Wang et al. 2025) ────────────────────
# #         #
# #         # F1 * improvement_ratio : how much total cost reduction did we achieve?
# #         #                          Primary training signal at episode end.
# #         # F2 * efficiency        : 1 - t/Tmax
# #         #                          Rewards finding good solutions early.
# #         #                          With fixed-length episodes efficiency = 0 always.
# #         #                          Becomes active if early-stopping is added later.
# #         #
# #         done = self.iteration >= self.max_iter
 
# #         if done:
# #             improvement = (self.init_cost - self.best.cost()) / self.init_cost
# #             efficiency  = 1.0 - (self.iteration / self.max_iter)
# #             reward     += F1 * improvement + F2 * efficiency
 
# #         return self._state(), float(reward), done, False, {}
 
# #     # ── State  (17 features) ───────────────────────────────────────────────
 
# #     def _state(self) -> np.ndarray:
# #         it        = self.iteration
# #         sp        = it / self.max_iter
# #         curr_cost = self.current.cost()
# #         best_cost = self.best.cost()
 
# #         # How far is current from best, normalised by init_cost
# #         delta = (curr_cost - best_cost) / max(self.init_cost, 1.0)
 
# #         # Operator usage rates
# #         total   = max(it, 1)
# #         d_usage = self.destroy_usage / total
# #         r_usage = self.repair_usage  / total
 
# #         # Instance features
# #         tws = [(o.pickup_node.due_time - o.pickup_node.ready_time)
# #                for o in self.instance.orders]
# #         avg_tw  = np.mean(tws) / MAX_TW
 
# #         svc = [o.pickup_node.service_time + o.delivery_node.service_time
# #                for o in self.instance.orders]
# #         avg_svc = np.mean(svc) / MAX_SERVICE
 
# #         fck_norm = self.instance.vehicle.fuel_cost_per_km / MAX_FUEL_COST
# #         vt_norm  = self.instance.vehicle.type_index / 3.0
 
# #         # Violation rate — fraction of orders with deadline violations
# #         # Gives PPO direct visibility into constraint satisfaction status
# #         n_orders  = max(len(self.instance.orders), 1)
# #         viol_rate = self.best.violations() / n_orders
 
# #         state = np.array([
# #             sp,                                          # [0]  search progress
# #             np.clip(delta,            -1.0, 1.0),       # [1]  current vs best gap
# #             np.clip(self.init_cost / MAX_COST, 0, 1),   # [2]  init cost (normalised)
# #             np.clip(best_cost      / MAX_COST, 0, 1),   # [3]  best cost (normalised)
# #             *np.clip(d_usage, 0.0, 1.0),                # [4-8]  destroy usage x5
# #             *np.clip(r_usage, 0.0, 1.0),                # [9-11] repair  usage x3
# #             np.clip(avg_tw,   0.0, 1.0),                # [12] avg time-window width
# #             np.clip(avg_svc,  0.0, 1.0),                # [13] avg service time
# #             np.clip(fck_norm, 0.0, 1.0),                # [14] fuel cost per km
# #             np.clip(vt_norm,  0.0, 1.0),                # [15] vehicle type
# #             np.clip(viol_rate, 0.0, 1.0),               # [16] deadline violation rate
# #         ], dtype=np.float32)
 
# #         return state



# """
# alns_env.py
# ===========
# Gymnasium environment for PPO-guided ALNS on PDVRPTW.
 
# State vector (17 features):
#   [0]  search_progress          (iteration / max_iter)
#   [1]  solution_delta           (current_cost - best_cost) / init_cost
#   [2]  init_cost_norm           (init_cost / MAX_COST)
#   [3]  best_cost_norm           (best_cost / MAX_COST)
#   [4-8] destroy_usage x5       (usage count / iteration, normalised)
#   [9-11] repair_usage x3       (usage count / iteration, normalised)
#   [12]  avg_tw_norm             (avg time window width / MAX_TW)
#   [13]  avg_service_norm        (avg service time / MAX_SERVICE)
#   [14]  fuel_cost_per_km_norm   (fuel_cost_per_km / MAX_FUEL_COST)
#   [15]  vehicle_type_enc        (0-3 normalised)
#   [16]  violation_rate          (deadline_violations / n_orders)
 
# Action: [destroy_index (0-4), repair_index (0-2)]
#   -> encoded as single integer: action = d * N_REPAIR + r
#   -> total 15 discrete actions
 
# Reward design (Wang et al. 2025, Eq. 3 / 4):
 
#   Step reward:
#     Case 1  new_cost < best_ever          -> +GAMMA * relative_improvement
#     Case 2  prev_cost > new_cost >= best  -> +ALPHA * delta / init_cost
#     Case 3  delta == 0                    ->  0
#     Case 4  new_cost > prev_cost          -> -BETA  * |delta| / init_cost
 
#   Exploration bonus:
#     +ETA * operator_diversity  (Shannon entropy of operator usage — fades as PPO converges)
 
#   Terminal reward (Eq. 4):
#     +F1 * (init_cost - best_cost) / init_cost   (quality)
#     +F2 * (1 - iteration / max_iter)            (efficiency — active if early-stopping added)
 
# Acceptance criterion:
#   Simulated Annealing — accepts worse solutions with probability
#   exp(-(new_cost - prev_cost) / temperature) where temperature
#   anneals from TEMP_START to TEMP_END over the episode.
#   This allows escape from local optima during training without
#   requiring any changes to the PPO policy itself.
# """
 
# import random
# import math
# import numpy as np
 
# try:
#     import gymnasium as gym
#     from gymnasium import spaces
# except ImportError:
#     import gym
#     from gym import spaces
 
# from typing import List, Dict, Optional
 
# from data_loader import Instance, Order, augment_instance
# from constraints import RouteNode, CheckResult, check_route, compute_metrics
# from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR
 
 
# # -----------------------------------------------------------------------------
# # Objective weights
# # -----------------------------------------------------------------------------
 
# W_TRAVEL_TIME = 1.0
# W_LATENESS    = 25.0      # strong penalty — violations were 105-140 in previous runs
# W_CARBON      = 0.05
# W_FUEL        = 0.1
# W_INFEASIBLE  = 1e6
 
# # -----------------------------------------------------------------------------
# # Step reward params  (Eq. 3 — Wang et al. 2025)
# # -----------------------------------------------------------------------------
# # ALPHA : reward scale for improving current solution (but not global best)
# # BETA  : penalty scale for worsening current solution (symmetric with ALPHA)
# # GAMMA : reward scale for beating global best — must be > ALPHA
# # ETA   : exploration bonus scale (operator diversity, fades as PPO converges)
 
# ALPHA = 1.0
# BETA  = 1.0
# GAMMA = 2.5
# ETA   = 0.3
 
# # -----------------------------------------------------------------------------
# # Terminal reward params  (Eq. 4 — Wang et al. 2025)
# # -----------------------------------------------------------------------------
# # F1 : weight on total improvement ratio  (primary goal)
# # F2 : weight on efficiency bonus         (secondary goal — active with early-stopping)
 
# F1 = 5.0
# F2 = 1.0
 
# # -----------------------------------------------------------------------------
# # Simulated Annealing acceptance params
# # -----------------------------------------------------------------------------
# # Temperature anneals linearly from TEMP_START -> TEMP_END over the episode.
# # High temperature early  -> accepts worse solutions often  (exploration)
# # Low  temperature late   -> rarely accepts worse solutions  (exploitation)
 
# TEMP_START = 1.0
# TEMP_END   = 0.01
 
# # -----------------------------------------------------------------------------
# # Normalisation ceilings
# # -----------------------------------------------------------------------------
 
# MAX_COST      = 1e5
# MAX_TW        = 600.0    # minutes (full working day)
# MAX_SERVICE   = 240.0    # minutes (2x max service time)
# MAX_FUEL_COST = 50.0     # INR/km ceiling
 
# # n_remove: number of orders to destroy each step
# MIN_REMOVE = 1
# MAX_REMOVE = 6
 
 
# # -----------------------------------------------------------------------------
# # Solution container
# # -----------------------------------------------------------------------------
 
# class Solution:
#     def __init__(self, route: List[RouteNode],
#                  orders_map: Dict[str, Order],
#                  vehicle):
#         self.route       = route
#         self.orders_map  = orders_map
#         self.vehicle     = vehicle
#         self._cost       = None
#         self._violations = None
 
#     def cost(self) -> float:
#         if self._cost is not None:
#             return self._cost
#         self._compute()
#         return self._cost
 
#     def violations(self) -> int:
#         if self._violations is not None:
#             return self._violations
#         self._compute()
#         return self._violations
 
#     def _compute(self):
#         """Single evaluation — cost and violations computed together."""
#         result  = check_route(self.route, self.orders_map, self.vehicle)
#         metrics = compute_metrics(self.route, self.vehicle)
 
#         c = (W_TRAVEL_TIME * metrics["travel_time_min"]
#            + W_LATENESS    * result.total_lateness
#            + W_CARBON      * metrics["carbon_emission_kg"]
#            + W_FUEL        * metrics["fuel_cost_inr"])
 
#         if not result.feasible:
#             c += W_INFEASIBLE
 
#         self._cost       = round(c, 4)
#         self._violations = getattr(result, "deadline_violations", 0)
 
#     def invalidate(self):
#         self._cost       = None
#         self._violations = None
 
#     def copy(self) -> "Solution":
#         return Solution(list(self.route), self.orders_map, self.vehicle)
 
 
# # -----------------------------------------------------------------------------
# # Environment
# # -----------------------------------------------------------------------------
 
# class ALNSEnv(gym.Env):
#     metadata = {"render_modes": []}
 
#     def __init__(self,
#                  instances: List[Instance],
#                  max_iter: int = 100,
#                  seed: int = 42):
#         super().__init__()
 
#         self.instances = instances
#         self.max_iter  = max_iter
#         self.rng       = random.Random(seed)
 
#         # 17 features, 15 discrete actions (5 destroy x 3 repair)
#         n_actions = N_DESTROY * N_REPAIR
#         self.action_space      = spaces.Discrete(n_actions)
#         self.observation_space = spaces.Box(
#             low=0.0, high=1.0, shape=(17,), dtype=np.float32
#         )
 
#         # Episode state (initialised in reset)
#         self.instance:    Optional[Instance] = None
#         self.orders_map:  Optional[Dict]     = None
#         self.current:     Optional[Solution] = None
#         self.best:        Optional[Solution] = None
#         self.init_cost:   float              = 1.0
#         self.iteration:   int                = 0
#         self.destroy_usage = np.zeros(N_DESTROY)
#         self.repair_usage  = np.zeros(N_REPAIR)
 
#     # ── Reset ──────────────────────────────────────────────────────────────
 
#     def reset(self, seed=None, options=None):
#         inst = self.rng.choice(self.instances)
#         self.instance   = augment_instance(inst, self.rng)
#         self.orders_map = {o.order_id: o for o in self.instance.orders}
 
#         initial_route  = self._build_initial_route()
#         self.current   = Solution(initial_route, self.orders_map, self.instance.vehicle)
#         self.best      = self.current.copy()
#         self.init_cost = max(self.current.cost(), 1.0)
 
#         self.iteration     = 0
#         self.destroy_usage = np.zeros(N_DESTROY)
#         self.repair_usage  = np.zeros(N_REPAIR)
 
#         return self._state(), {}
 
#     def _build_initial_route(self) -> List[RouteNode]:
#         route = []
#         for o in self.instance.orders:
#             route.append(RouteNode(o.order_id, o.pickup_node))
#             route.append(RouteNode(o.order_id, o.delivery_node))
#         return route
 
#     # ── Step ───────────────────────────────────────────────────────────────
 
#     def step(self, action: int):
#         d_idx = min(action // N_REPAIR, N_DESTROY - 1)
#         r_idx = min(action  % N_REPAIR, N_REPAIR  - 1)
 
#         destroy_op = DESTROY_OPS[d_idx]
#         repair_op  = REPAIR_OPS[r_idx]
 
#         n_remove         = self.rng.randint(MIN_REMOVE, min(MAX_REMOVE, len(self.instance.orders)))
#         prev_cost        = self.current.cost()
#         best_cost_before = self.best.cost()
 
#         # ── Destroy ────────────────────────────────────────────────────────
#         new_route, removed = destroy_op(
#             self.current.route, self.orders_map,
#             self.instance.vehicle, n_remove, self.rng
#         )
 
#         # ── Repair ─────────────────────────────────────────────────────────
#         new_route = repair_op(
#             new_route, removed, self.orders_map,
#             self.instance.vehicle, self.rng
#         )
 
#         candidate = Solution(new_route, self.orders_map, self.instance.vehicle)
#         new_cost  = candidate.cost()
 
#         # ── Simulated Annealing acceptance ─────────────────────────────────
#         # Temperature anneals linearly from TEMP_START to TEMP_END.
#         # Early in episode: high temp -> frequently accepts worse solutions.
#         # Late  in episode: low  temp -> almost never accepts worse solutions.
#         # This provides structured exploration that greedy acceptance cannot.
#         progress    = self.iteration / max(self.max_iter, 1)
#         temperature = TEMP_START * (1.0 - progress) + TEMP_END
 
#         if new_cost < prev_cost:
#             # Always accept improvements
#             self.current = candidate
#         else:
#             # Accept worse solution with SA probability
#             sa_prob = math.exp(-(new_cost - prev_cost) / max(temperature, 1e-9))
#             if self.rng.random() < sa_prob:
#                 self.current = candidate
 
#         # Update global best regardless of acceptance
#         if new_cost < self.best.cost():
#             self.best = candidate.copy()
 
#         # ── Update usage counters ───────────────────────────────────────────
#         self.destroy_usage[d_idx] += 1
#         self.repair_usage[r_idx]  += 1
#         self.iteration += 1
 
#         # ── Step reward  (Eq. 3 — Wang et al. 2025) ────────────────────────
#         # Priority order:
#         #   1. Beat global best -> GAMMA  (strongest signal, primary goal)
#         #   2. Improve current  -> ALPHA
#         #   3. No change        -> 0
#         #   4. Worsen current   -> -BETA
#         #
#         # Note: reward is based on cost delta vs prev_cost, not on acceptance.
#         # PPO learns which operators produce good candidates — the SA acceptance
#         # is separate and does not distort the reward signal.
 
#         delta = prev_cost - new_cost
 
#         if new_cost < best_cost_before:
#             reward = GAMMA * (best_cost_before - new_cost) / max(best_cost_before, 1.0)
#         elif delta > 0:
#             reward = ALPHA * delta / self.init_cost
#         elif delta == 0:
#             reward = 0.0
#         else:
#             reward = -BETA * abs(delta) / self.init_cost
 
#         # ── Exploration bonus ───────────────────────────────────────────────
#         # Shannon entropy of operator usage distribution.
#         # Rewards PPO for trying diverse operator combinations.
#         # Max bonus = ETA when all operators used equally.
#         # Fades to 0 as PPO specialises onto best operators.
 
#         all_usage = np.concatenate([self.destroy_usage, self.repair_usage])
#         total_use = all_usage.sum()
#         if total_use > 0:
#             probs       = all_usage / total_use
#             probs       = probs[probs > 0]
#             entropy     = -np.sum(probs * np.log(probs))
#             max_entropy = np.log(N_DESTROY + N_REPAIR)
#             diversity   = entropy / max_entropy
#             reward     += ETA * diversity
 
#         # ── Terminal reward  (Eq. 4 — Wang et al. 2025) ────────────────────
#         done = self.iteration >= self.max_iter
 
#         if done:
#             improvement = (self.init_cost - self.best.cost()) / self.init_cost
#             efficiency  = 1.0 - (self.iteration / self.max_iter)
#             reward     += F1 * improvement + F2 * efficiency
 
#         return self._state(), float(reward), done, False, {}
 
#     # ── State  (17 features) ───────────────────────────────────────────────
 
#     def _state(self) -> np.ndarray:
#         it        = self.iteration
#         sp        = it / self.max_iter
#         curr_cost = self.current.cost()
#         best_cost = self.best.cost()
 
#         delta   = (curr_cost - best_cost) / max(self.init_cost, 1.0)
#         total   = max(it, 1)
#         d_usage = self.destroy_usage / total
#         r_usage = self.repair_usage  / total
 
#         tws = [(o.pickup_node.due_time - o.pickup_node.ready_time)
#                for o in self.instance.orders]
#         avg_tw  = np.mean(tws) / MAX_TW
 
#         svc = [o.pickup_node.service_time + o.delivery_node.service_time
#                for o in self.instance.orders]
#         avg_svc = np.mean(svc) / MAX_SERVICE
 
#         fck_norm  = self.instance.vehicle.fuel_cost_per_km / MAX_FUEL_COST
#         vt_norm   = self.instance.vehicle.type_index / 3.0
 
#         # Violation rate — direct constraint satisfaction signal for PPO
#         n_orders  = max(len(self.instance.orders), 1)
#         viol_rate = self.best.violations() / n_orders
 
#         state = np.array([
#             sp,                                          # [0]  search progress
#             np.clip(delta,            -1.0, 1.0),       # [1]  current vs best gap
#             np.clip(self.init_cost / MAX_COST, 0, 1),   # [2]  init cost normalised
#             np.clip(best_cost      / MAX_COST, 0, 1),   # [3]  best cost normalised
#             *np.clip(d_usage, 0.0, 1.0),                # [4-8]  destroy usage x5
#             *np.clip(r_usage, 0.0, 1.0),                # [9-11] repair  usage x3
#             np.clip(avg_tw,    0.0, 1.0),               # [12] avg time-window width
#             np.clip(avg_svc,   0.0, 1.0),               # [13] avg service time
#             np.clip(fck_norm,  0.0, 1.0),               # [14] fuel cost per km
#             np.clip(vt_norm,   0.0, 1.0),               # [15] vehicle type
#             np.clip(viol_rate, 0.0, 1.0),               # [16] deadline violation rate
#         ], dtype=np.float32)
 
#         return state


# import json
# import os
# from presolve import route_from_json  # cache loader

import json
import os
from presolve import route_from_json
from constants import W_TRAVEL_TIME, W_LATENESS, W_CARBON, W_FUEL, W_INFEASIBLE

"""
alns_env.py
===========
Gymnasium environment for PPO-guided ALNS on PDVRPTW.

State vector (17 features):
  [0]  search_progress          (iteration / max_iter)
  [1]  solution_delta           (current_cost - best_cost) / init_cost
  [2]  init_cost_norm           (init_cost / MAX_COST)
  [3]  best_cost_norm           (best_cost / MAX_COST)
  [4-8] destroy_usage x5       (usage count / iteration, normalised)
  [9-11] repair_usage x3       (usage count / iteration, normalised)
  [12]  avg_tw_norm             (avg time window width / MAX_TW)
  [13]  avg_service_norm        (avg service time / MAX_SERVICE)
  [14]  fuel_cost_per_km_norm   (fuel_cost_per_km / MAX_FUEL_COST)
  [15]  vehicle_type_enc        (0-3 normalised)
  [16]  violation_rate          (deadline_violations / n_orders)

Action: [destroy_index (0-4), repair_index (0-2)]
  -> encoded as single integer: action = d * N_REPAIR + r
  -> total 15 discrete actions

Reward design (Wang et al. 2025, Eq. 3 / 4):
  Step reward:
    Case 1  new_cost < best_ever          -> +GAMMA * relative_improvement
    Case 2  prev_cost > new_cost >= best  -> +ALPHA * delta / init_cost
    Case 3  delta == 0                    ->  0
    Case 4  new_cost > prev_cost          -> -BETA  * |delta| / init_cost
  Exploration bonus:
    +ETA * operator_diversity
  Terminal reward (Eq. 4):
    +F1 * (init_cost - best_cost) / init_cost
    +F2 * (1 - iteration / max_iter)

Acceptance: Simulated Annealing (TEMP_START -> TEMP_END over episode)

Initial solution: time-window sorted greedy insertion
  Orders sorted by pickup due_time (earliest deadline first).
  Each order inserted at the cheapest position in the current partial route.
  This produces a much better starting point than sequential appending,
  dramatically reducing initial lateness.
"""

import random
import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from typing import List, Dict, Optional

from data_loader import Instance, Order, augment_instance, get_travel_time
from constraints import RouteNode, CheckResult, check_route, compute_metrics
from alns_operators import DESTROY_OPS, REPAIR_OPS, N_DESTROY, N_REPAIR


# -----------------------------------------------------------------------------
# Objective weights
# -----------------------------------------------------------------------------

# W_TRAVEL_TIME = 1.0
# W_LATENESS    = 500.0
# W_CARBON      = 0.05
# W_FUEL        = 0.1
# W_INFEASIBLE  = 1e7

# -----------------------------------------------------------------------------
# Step reward params  (Eq. 3 — Wang et al. 2025)
# -----------------------------------------------------------------------------

ALPHA = 1.0
BETA  = 1.0
GAMMA = 2.5
ETA   = 0.0

# -----------------------------------------------------------------------------
# Terminal reward params  (Eq. 4 — Wang et al. 2025)
# -----------------------------------------------------------------------------

F1 = 5.0
F2 = 1.0

# -----------------------------------------------------------------------------
# Simulated Annealing acceptance params
# -----------------------------------------------------------------------------

# TEMP_START = 200.0
# TEMP_END   = 2.0

# TEMP_START = 10000.0
# TEMP_END   = 50.0

TEMP_START = 500.0
TEMP_END   = 5.0

# -----------------------------------------------------------------------------
# Normalisation ceilings
# -----------------------------------------------------------------------------

MAX_COST      = 1e5      # raised from 1e5 — actual costs are in millions due to lateness
MAX_TW        = 600.0
MAX_SERVICE   = 240.0
MAX_FUEL_COST = 50.0

# n_remove: number of orders to destroy each step
MIN_REMOVE = 1
MAX_REMOVE = 6


OR_CACHE_DIR = "data/or_cache"   # same path you used in presolve.py

# -----------------------------------------------------------------------------
# Solution container
# -----------------------------------------------------------------------------

class Solution:
    def __init__(self, route: List[RouteNode],
                 orders_map: Dict[str, Order],
                 vehicle):
        self.route       = route
        self.orders_map  = orders_map
        self.vehicle     = vehicle
        self._cost       = None
        self._violations = None

    def cost(self) -> float:
        if self._cost is not None:
            return self._cost
        self._compute()
        return self._cost

    def violations(self) -> int:
        if self._violations is not None:
            return self._violations
        self._compute()
        return self._violations

    def _compute(self):
        result  = check_route(self.route, self.orders_map, self.vehicle)
        metrics = compute_metrics(self.route, self.vehicle)

        c = (W_TRAVEL_TIME * metrics["travel_time_min"]
           + W_LATENESS    * result.total_lateness
           + W_CARBON      * metrics["carbon_emission_kg"]
           + W_FUEL        * metrics["fuel_cost_inr"])

        if not result.feasible:
            c += W_INFEASIBLE

        self._cost       = round(c, 4)
        self._violations = getattr(result, "deadline_violations", 0)

    def invalidate(self):
        self._cost       = None
        self._violations = None

    def copy(self) -> "Solution":
        return Solution(list(self.route), self.orders_map, self.vehicle)


# -----------------------------------------------------------------------------
# Initial route construction — time-window sorted greedy insertion
# -----------------------------------------------------------------------------

def build_tw_sorted_route(orders: List[Order],
                           orders_map: Dict[str, Order]) -> List[RouteNode]:
    """
    Build an initial route by inserting orders in earliest-deadline-first order.
    Each order is inserted at the position that minimises cumulative travel time.

    This is far better than sequential appending because it respects time windows
    from the start, reducing initial lateness by orders of magnitude.

    Steps:
      1. Sort orders by pickup due_time (tightest deadline first)
      2. For each order, try all valid (pickup_pos, delivery_pos) pairs
         where delivery_pos > pickup_pos
      3. Insert at the pair with minimum added travel time
      4. Fall back to appending at end if route is empty
    """
    sorted_orders = sorted(orders, key=lambda o: o.pickup_node.due_time)
    route: List[RouteNode] = []

    for order in sorted_orders:
        if not route:
            route.append(RouteNode(order.order_id, order.pickup_node))
            route.append(RouteNode(order.order_id, order.delivery_node))
            continue

        n = len(route)
        best_cost = float('inf')
        best_i, best_j = 0, 1

        # Try all valid insertion positions (capped for speed)
        limit = min(n + 1, 25)
        for i in range(limit):
            for j in range(i + 1, min(limit + 1, n + 2)):
                # Estimate added travel cost of inserting pickup at i, delivery at j
                trial = list(route)
                trial.insert(i, RouteNode(order.order_id, order.pickup_node))
                trial.insert(j, RouteNode(order.order_id, order.delivery_node))

                # Fast travel time estimate
                tt = 0.0
                for k in range(len(trial) - 1):
                    tt += get_travel_time(trial[k].node, trial[k+1].node)

                # Lateness estimate
                time_now = 0.0
                lateness = 0.0
                prev = None
                for rn in trial:
                    node = rn.node
                    if prev is not None:
                        time_now += get_travel_time(prev, node)
                    time_now = max(time_now, node.ready_time)
                    if time_now > node.due_time:
                        lateness += time_now - node.due_time
                    time_now += node.service_time
                    prev = node

                cost = tt + W_LATENESS * lateness
                if cost < best_cost:
                    best_cost = cost
                    best_i, best_j = i, j

        route.insert(best_i, RouteNode(order.order_id, order.pickup_node))
        route.insert(best_j, RouteNode(order.order_id, order.delivery_node))

    return route


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class ALNSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 instances: List[Instance],
                 max_iter: int = 100,
                 seed: int = 42):
        super().__init__()

        self.instances = instances
        self.max_iter  = max_iter
        self.rng       = random.Random(seed)

        n_actions = N_DESTROY * N_REPAIR
        self.action_space      = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        self.instance:    Optional[Instance] = None
        self.orders_map:  Optional[Dict]     = None
        self.current:     Optional[Solution] = None
        self.best:        Optional[Solution] = None
        self.init_cost:   float              = 1.0
        self.iteration:   int                = 0
        self.destroy_usage = np.zeros(N_DESTROY)
        self.repair_usage  = np.zeros(N_REPAIR)

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.best_found_at = 0
        inst = self.rng.choice(self.instances)
        self.instance   = augment_instance(inst, self.rng)
        self.orders_map = {o.order_id: o for o in self.instance.orders}

        #initial_route  = build_tw_sorted_route(self.instance.orders, self.orders_map)
        initial_route = self._load_cached_route() or \
                        build_tw_sorted_route(self.instance.orders, self.orders_map)
        self.current   = Solution(initial_route, self.orders_map, self.instance.vehicle)
        self.best      = self.current.copy()
        self.init_cost = max(self.current.cost(), 1.0)

        self.iteration     = 0
        self.destroy_usage = np.zeros(N_DESTROY)
        self.repair_usage  = np.zeros(N_REPAIR)

        return self._state(), {}

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: int):
        d_idx = min(action // N_REPAIR, N_DESTROY - 1)
        r_idx = min(action  % N_REPAIR, N_REPAIR  - 1)

        destroy_op = DESTROY_OPS[d_idx]
        repair_op  = REPAIR_OPS[r_idx]

        n_remove         = self.rng.randint(MIN_REMOVE, min(MAX_REMOVE, len(self.instance.orders)))
        prev_cost        = self.current.cost()
        best_cost_before = self.best.cost()

        new_route, removed = destroy_op(
            self.current.route, self.orders_map,
            self.instance.vehicle, n_remove, self.rng
        )
        new_route = repair_op(
            new_route, removed, self.orders_map,
            self.instance.vehicle, self.rng
        )

        candidate = Solution(new_route, self.orders_map, self.instance.vehicle)
        new_cost  = candidate.cost()

        # Simulated Annealing acceptance
        progress    = self.iteration / max(self.max_iter, 1)
        temperature = TEMP_START * (1.0 - progress) + TEMP_END

        if new_cost < prev_cost:
            self.current = candidate
        else:
            sa_prob = math.exp(-(new_cost - prev_cost) / max(temperature, 1e-9))
            if self.rng.random() < sa_prob:
                self.current = candidate

        if new_cost < self.best.cost():
            self.best = candidate.copy()
            self.best_found_at = self.iteration

        self.destroy_usage[d_idx] += 1
        self.repair_usage[r_idx]  += 1
        self.iteration += 1

        # Step reward (Eq. 3)
        # delta = prev_cost - new_cost

        # if new_cost < best_cost_before:
        #     reward = GAMMA * (best_cost_before - new_cost) / max(best_cost_before, 1.0)
        # elif delta > 0:
        #     reward = ALPHA * delta / self.init_cost
        # elif delta == 0:
        #     reward = 0.0
        # else:
        #     reward = -BETA * abs(delta) / self.init_cost
        # Step reward (Eq. 3)
        delta = prev_cost - new_cost

        if new_cost < best_cost_before:
            reward = GAMMA * (best_cost_before - new_cost) / max(best_cost_before, 1.0)
        elif delta > 0:
            reward = ALPHA * delta / self.init_cost
        elif delta == 0:
            reward = 0.0
        else:
            reward = -BETA * abs(delta) / self.init_cost

        # CLIP reward so PPO never sees values outside [-10, 10]
        reward = float(np.clip(reward, -10.0, 10.0))

        # Exploration bonus
        all_usage = np.concatenate([self.destroy_usage, self.repair_usage])
        total_use = all_usage.sum()
        if total_use > 0:
            probs       = all_usage / total_use
            probs       = probs[probs > 0]
            entropy     = -np.sum(probs * np.log(probs))
            max_entropy = np.log(N_DESTROY + N_REPAIR)
            diversity   = entropy / max_entropy
            reward     += ETA * diversity

        # Terminal reward (Eq. 4)
        done = self.iteration >= self.max_iter

        # if done:
        #     improvement = (self.init_cost - self.best.cost()) / self.init_cost
        #     efficiency  = 1.0 - (self.iteration / self.max_iter)
        #     reward     += F1 * improvement + F2 * efficiency
        if done:
            improvement = (self.init_cost - self.best.cost()) / self.init_cost
            efficiency  = 1.0 - (self.best_found_at / self.max_iter)
            reward     += F1 * improvement + F2 * efficiency

        return self._state(), float(reward), done, False, {}

    # ── State (17 features) ────────────────────────────────────────────────

    def _state(self) -> np.ndarray:
        it        = self.iteration
        sp        = it / self.max_iter
        curr_cost = self.current.cost()
        best_cost = self.best.cost()

        delta   = (curr_cost - best_cost) / max(self.init_cost, 1.0)
        total   = max(it, 1)
        d_usage = self.destroy_usage / total
        r_usage = self.repair_usage  / total

        tws = [(o.pickup_node.due_time - o.pickup_node.ready_time)
               for o in self.instance.orders]
        avg_tw  = np.mean(tws) / MAX_TW

        svc = [o.pickup_node.service_time + o.delivery_node.service_time
               for o in self.instance.orders]
        avg_svc = np.mean(svc) / MAX_SERVICE

        fck_norm  = self.instance.vehicle.fuel_cost_per_km / MAX_FUEL_COST
        vt_norm   = self.instance.vehicle.type_index / 3.0

        n_orders  = max(len(self.instance.orders), 1)
        viol_rate = self.best.violations() / n_orders

        state = np.array([
            sp,
            np.clip(delta,            -1.0, 1.0),
            np.clip(self.init_cost / MAX_COST, 0, 1),
            np.clip(best_cost      / MAX_COST, 0, 1),
            *np.clip(d_usage, 0.0, 1.0),
            *np.clip(r_usage, 0.0, 1.0),
            np.clip(avg_tw,    0.0, 1.0),
            np.clip(avg_svc,   0.0, 1.0),
            np.clip(fck_norm,  0.0, 1.0),
            np.clip(vt_norm,   0.0, 1.0),
            np.clip(viol_rate, 0.0, 1.0),
        ], dtype=np.float32)

        return state


    def _load_cached_route(self) -> Optional[List[RouteNode]]:
        """Load OR-Tools warm-start from cache. Returns None if not found."""
        cache_path = os.path.join(OR_CACHE_DIR, f"{self.instance.instance_id}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return route_from_json(data["route"], self.orders_map)
        except Exception:
            return None