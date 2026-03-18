# """
# visualizer.py
# =============
# All visual outputs for PPO-ALNS:
#   1. Training curves (reward, cost improvement, violations per batch)
#   2. Route map (GPS scatter of pickup/delivery nodes + route lines)
#   3. Cost breakdown pie chart
#   4. Operator usage bar chart

# Usage:
#     from visualizer import plot_training_log, plot_route, plot_cost_breakdown
# """

# import os
# import json
# import matplotlib
# matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# from typing import List, Dict, Optional


# OUTPUT_DIR = "outputs"

# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Training curves
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_training_log(log_path: str, batch_label: str = ""):
#     """
#     Read a training log CSV/JSON and plot:
#       - Episode reward over time
#       - Best cost improvement % over time
#       - Deadline violations over time

#     log_path: path to training_log.json written by BatchTrainer
#     """
#     if not os.path.exists(log_path):
#         print(f"[VIZ] Log not found: {log_path}")
#         return

#     with open(log_path) as f:
#         log = json.load(f)

#     episodes    = [e["episode"]     for e in log]
#     rewards     = [e["reward"]      for e in log]
#     costs       = [e["best_cost"]   for e in log]
#     violations  = [e["violations"]  for e in log]
#     init_costs  = [e["init_cost"]   for e in log]

#     # Improvement % = (init - best) / init * 100
#     improvements = [
#         (ic - bc) / max(ic, 1) * 100
#         for ic, bc in zip(init_costs, costs)
#     ]

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle(f"PPO-ALNS Training Progress {batch_label}", fontsize=14, fontweight="bold")

#     # Reward
#     axes[0].plot(episodes, rewards, color="#2196F3", linewidth=0.8, alpha=0.6, label="Episode reward")
#     axes[0].plot(episodes, _smooth(rewards, 20), color="#0D47A1", linewidth=2, label="Smoothed (20-ep)")
#     axes[0].set_ylabel("Reward")
#     axes[0].legend(fontsize=8)
#     axes[0].grid(True, alpha=0.3)
#     axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

#     # Cost improvement
#     axes[1].plot(episodes, improvements, color="#4CAF50", linewidth=0.8, alpha=0.6)
#     axes[1].plot(episodes, _smooth(improvements, 20), color="#1B5E20", linewidth=2, label="Cost improvement %")
#     axes[1].set_ylabel("Improvement %")
#     axes[1].legend(fontsize=8)
#     axes[1].grid(True, alpha=0.3)

#     # Violations
#     axes[2].plot(episodes, violations, color="#F44336", linewidth=0.8, alpha=0.6)
#     axes[2].plot(episodes, _smooth(violations, 20), color="#B71C1C", linewidth=2, label="Deadline violations")
#     axes[2].set_ylabel("Violations")
#     axes[2].set_xlabel("Episode")
#     axes[2].legend(fontsize=8)
#     axes[2].grid(True, alpha=0.3)

#     plt.tight_layout()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, f"training_curves{batch_label.replace(' ','_')}.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[VIZ] Training curves saved → {path}")
#     return path


# def _smooth(values: List[float], window: int) -> List[float]:
#     """Moving average smoothing."""
#     if len(values) < window:
#         return values
#     result = []
#     for i in range(len(values)):
#         start = max(0, i - window // 2)
#         end   = min(len(values), i + window // 2 + 1)
#         result.append(np.mean(values[start:end]))
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. Route map
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_route(instance, solution, title: str = ""):
#     """
#     Plot the route on a Delhi NCR coordinate map.
#     Blue dots = pickup, Red dots = delivery, Green star = depot.
#     Lines show the route sequence.
#     """
#     from constraints import RouteNode

#     route = solution.route
#     orders_map = {o.order_id: o for o in instance.orders}

#     # Collect coordinates
#     depot_lat = instance.depot.lat
#     depot_lon = instance.depot.lon

#     pickup_lats, pickup_lons, pickup_labels     = [], [], []
#     delivery_lats, delivery_lons, delivery_labels = [], [], []
#     route_lats, route_lons = [depot_lat], [depot_lon]

#     for rn in route:
#         n = rn.node
#         route_lats.append(n.lat)
#         route_lons.append(n.lon)
#         if rn.is_pickup:
#             pickup_lats.append(n.lat)
#             pickup_lons.append(n.lon)
#             pickup_labels.append(rn.order_id[-3:])
#         else:
#             delivery_lats.append(n.lat)
#             delivery_lons.append(n.lon)
#             delivery_labels.append(rn.order_id[-3:])

#     fig, ax = plt.subplots(figsize=(14, 10))

#     # Route lines
#     ax.plot(route_lons, route_lats,
#             color="#90CAF9", linewidth=0.8, alpha=0.6, zorder=1)

#     # Arrows every 5 stops to show direction
#     for i in range(0, len(route_lons) - 1, 5):
#         dx = route_lons[i+1] - route_lons[i]
#         dy = route_lats[i+1]  - route_lats[i]
#         ax.annotate("", xy=(route_lons[i+1], route_lats[i+1]),
#                     xytext=(route_lons[i], route_lats[i]),
#                     arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8),
#                     zorder=2)

#     # Pickup nodes
#     ax.scatter(pickup_lons, pickup_lats,
#                c="#2196F3", s=60, zorder=4, label="Pickup", marker="o", edgecolors="white", linewidths=0.5)

#     # Delivery nodes
#     ax.scatter(delivery_lons, delivery_lats,
#                c="#F44336", s=60, zorder=4, label="Delivery", marker="s", edgecolors="white", linewidths=0.5)

#     # Depot
#     ax.scatter([depot_lon], [depot_lat],
#                c="#4CAF50", s=200, zorder=5, label="Depot", marker="*", edgecolors="black", linewidths=0.8)

#     # Labels for first 20 stops only (avoid clutter)
#     for i, (la, lo, lbl) in enumerate(zip(pickup_lats[:20], pickup_lons[:20], pickup_labels[:20])):
#         ax.annotate(lbl, (lo, la), fontsize=5, ha="center", va="bottom",
#                     color="#0D47A1", xytext=(0, 3), textcoords="offset points")

#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     title_str = title or f"Route — {instance.instance_id}"
#     ax.set_title(f"{title_str}\n{len(route)} stops | {len(instance.orders)} orders | "
#                  f"{instance.vehicle.vehicle_type} ({instance.vehicle.fuel_type})",
#                  fontsize=11)
#     ax.legend(loc="upper right", fontsize=9)
#     ax.grid(True, alpha=0.2)

#     plt.tight_layout()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, f"{instance.instance_id}_route_map.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[VIZ] Route map saved → {path}")
#     return path


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. Cost breakdown pie chart
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_cost_breakdown(metrics: Dict, lateness: float, instance_id: str = ""):
#     """
#     Pie chart showing contribution of each cost component.
#     """
#     from alns_env import W_TRAVEL_TIME, W_LATENESS, W_CARBON, W_FUEL

#     travel_cost  = W_TRAVEL_TIME * metrics["travel_time_min"]
#     late_cost    = W_LATENESS    * lateness
#     carbon_cost  = W_CARBON      * metrics["carbon_emission_kg"]
#     fuel_cost    = W_FUEL        * metrics["fuel_cost_inr"]

#     labels = ["Travel Time", "Lateness Penalty", "Carbon Emission", "Fuel Cost"]
#     values = [travel_cost, late_cost, carbon_cost, fuel_cost]
#     colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

#     # Remove zero slices
#     filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
#     if not filtered:
#         return
#     labels, values, colors = zip(*filtered)

#     fig, ax = plt.subplots(figsize=(8, 6))
#     wedges, texts, autotexts = ax.pie(
#         values, labels=labels, colors=colors,
#         autopct="%1.1f%%", startangle=140,
#         wedgeprops=dict(edgecolor="white", linewidth=1.5)
#     )
#     for at in autotexts:
#         at.set_fontsize(9)

#     ax.set_title(f"Objective Cost Breakdown\n{instance_id}", fontsize=11)

#     plt.tight_layout()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, f"{instance_id}_cost_breakdown.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[VIZ] Cost breakdown saved → {path}")
#     return path


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. Operator usage bar chart
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_operator_usage(destroy_counts: List[int], repair_counts: List[int],
#                         instance_id: str = ""):
#     """
#     Bar chart of how many times each destroy and repair operator was chosen.
#     """
#     destroy_names = ["Random", "Worst Cost", "Shaw", "String", "Geo Segment"]
#     repair_names  = ["Greedy", "Criticality", "Regret-2"]

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle(f"Operator Usage — {instance_id}", fontsize=12)

#     ax1.bar(destroy_names, destroy_counts, color="#2196F3", edgecolor="white")
#     ax1.set_title("Destroy Operators")
#     ax1.set_ylabel("Times Selected")
#     ax1.tick_params(axis="x", rotation=15)
#     for i, v in enumerate(destroy_counts):
#         ax1.text(i, v + 0.5, str(v), ha="center", fontsize=9)

#     ax2.bar(repair_names, repair_counts, color="#4CAF50", edgecolor="white")
#     ax2.set_title("Repair Operators")
#     ax2.set_ylabel("Times Selected")
#     for i, v in enumerate(repair_counts):
#         ax2.text(i, v + 0.5, str(v), ha="center", fontsize=9)

#     plt.tight_layout()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, f"{instance_id}_operator_usage.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[VIZ] Operator usage saved → {path}")
#     return path


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. Batch summary chart — compare all batches
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_batch_summary(batch_logs: List[Dict]):
#     """
#     After all batches complete, plot:
#       - Final avg reward per batch
#       - Final avg improvement % per batch
#       - Final avg violations per batch

#     batch_logs: list of {"batch": int, "avg_reward": float,
#                           "avg_improvement": float, "avg_violations": float}
#     """
#     if not batch_logs:
#         return

#     batches = [b["batch"] for b in batch_logs]
#     rewards = [b["avg_reward"]      for b in batch_logs]
#     improv  = [b["avg_improvement"] for b in batch_logs]
#     viols   = [b["avg_violations"]  for b in batch_logs]

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     fig.suptitle("Batch Training Summary", fontsize=14, fontweight="bold")

#     axes[0].bar(batches, rewards, color="#2196F3")
#     axes[0].set_title("Avg Reward per Batch")
#     axes[0].set_xlabel("Batch")
#     axes[0].set_ylabel("Avg Reward")

#     axes[1].bar(batches, improv, color="#4CAF50")
#     axes[1].set_title("Avg Cost Improvement %")
#     axes[1].set_xlabel("Batch")
#     axes[1].set_ylabel("Improvement %")

#     axes[2].bar(batches, viols, color="#F44336")
#     axes[2].set_title("Avg Deadline Violations")
#     axes[2].set_xlabel("Batch")
#     axes[2].set_ylabel("Violations")

#     plt.tight_layout()
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     path = os.path.join(OUTPUT_DIR, "batch_summary.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[VIZ] Batch summary saved → {path}")
#     return path



"""
visualizer.py
=============
All visual outputs for PPO-ALNS:
  1. Training curves (reward, cost improvement, violations per episode)
  2. Batch summary chart (trend lines across all batches)
  3. Training summary printout (console)
  4. Route map
  5. Cost breakdown pie chart
  6. Operator usage bar chart
"""
 
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict, Optional
 
 
OUTPUT_DIR = "outputs"
 
 
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
 
def _smooth(values: List[float], window: int) -> List[float]:
    """Moving average smoothing."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end   = min(len(values), i + window // 2 + 1)
        result.append(np.mean(values[start:end]))
    return result
 
 
def _trend_line(x, y):
    """Linear regression trend line."""
    if len(x) < 2:
        return y
    coeffs = np.polyfit(x, y, 1)
    return np.polyval(coeffs, x)
 
 
# -----------------------------------------------------------------------------
# 1. Training curves (per episode)
# -----------------------------------------------------------------------------
 
def plot_training_log(log_path: str, batch_label: str = ""):
    if not os.path.exists(log_path):
        print(f"[VIZ] Log not found: {log_path}")
        return
 
    with open(log_path) as f:
        log = json.load(f)
 
    episodes   = [e["episode"]    for e in log]
    rewards    = [e["reward"]     for e in log]
    costs      = [e["best_cost"]  for e in log]
    violations = [e["violations"] for e in log]
    init_costs = [e["init_cost"]  for e in log]
 
    improvements = [
        (ic - bc) / max(ic, 1) * 100
        for ic, bc in zip(init_costs, costs)
    ]
 
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"PPO-ALNS Training Progress {batch_label}", fontsize=14, fontweight="bold")
 
    # Reward
    axes[0].plot(episodes, rewards, color="#2196F3", linewidth=0.6, alpha=0.4, label="Episode reward")
    axes[0].plot(episodes, _smooth(rewards, 20), color="#0D47A1", linewidth=2, label="Smoothed (20-ep)")
    axes[0].plot(episodes, _trend_line(episodes, rewards), color="#FF5722", linewidth=1.5,
                 linestyle="--", label="Trend")
    axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
 
    # Improvement %
    axes[1].plot(episodes, improvements, color="#4CAF50", linewidth=0.6, alpha=0.4)
    axes[1].plot(episodes, _smooth(improvements, 20), color="#1B5E20", linewidth=2, label="Cost improvement %")
    axes[1].plot(episodes, _trend_line(episodes, improvements), color="#FF5722", linewidth=1.5,
                 linestyle="--", label="Trend")
    axes[1].set_ylabel("Improvement %")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
 
    # Violations
    axes[2].plot(episodes, violations, color="#F44336", linewidth=0.6, alpha=0.4)
    axes[2].plot(episodes, _smooth(violations, 20), color="#B71C1C", linewidth=2, label="Deadline violations")
    axes[2].plot(episodes, _trend_line(episodes, violations), color="#FF5722", linewidth=1.5,
                 linestyle="--", label="Trend")
    axes[2].set_ylabel("Violations")
    axes[2].set_xlabel("Episode")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
 
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"training_curves{batch_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Training curves saved → {path}")
    return path
 
 
# -----------------------------------------------------------------------------
# 2. Batch summary chart (trend lines — much more readable than bars)
# -----------------------------------------------------------------------------
 
def plot_batch_summary(batch_logs: List[Dict]):
    if not batch_logs:
        return
 
    batches = [b["batch"]           for b in batch_logs]
    rewards = [b["avg_reward"]      for b in batch_logs]
    improv  = [b["avg_improvement"] for b in batch_logs]
    viols   = [b["avg_violations"]  for b in batch_logs]
 
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("PPO-ALNS Batch Training Summary", fontsize=14, fontweight="bold")
 
    # ── Reward ──
    axes[0].plot(batches, rewards, color="#2196F3", linewidth=1.2, alpha=0.5, marker="o",
                 markersize=3, label="Avg reward")
    axes[0].plot(batches, _smooth(rewards, 10), color="#0D47A1", linewidth=2.5, label="Smoothed")
    axes[0].plot(batches, _trend_line(batches, rewards), color="#FF5722", linewidth=1.8,
                 linestyle="--", label="Trend")
    axes[0].set_ylabel("Avg Reward", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
 
    # Annotate start and end
    axes[0].annotate(f"{rewards[0]:.2f}", (batches[0], rewards[0]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color="#0D47A1")
    axes[0].annotate(f"{rewards[-1]:.2f}", (batches[-1], rewards[-1]),
                     textcoords="offset points", xytext=(-25, 5), fontsize=8, color="#0D47A1")
 
    # ── Improvement % ──
    axes[1].plot(batches, improv, color="#4CAF50", linewidth=1.2, alpha=0.5, marker="o",
                 markersize=3, label="Avg improvement %")
    axes[1].plot(batches, _smooth(improv, 10), color="#1B5E20", linewidth=2.5, label="Smoothed")
    axes[1].plot(batches, _trend_line(batches, improv), color="#FF5722", linewidth=1.8,
                 linestyle="--", label="Trend")
    axes[1].set_ylabel("Improvement %", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
 
    axes[1].annotate(f"{improv[0]:.1f}%", (batches[0], improv[0]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color="#1B5E20")
    axes[1].annotate(f"{improv[-1]:.1f}%", (batches[-1], improv[-1]),
                     textcoords="offset points", xytext=(-30, 5), fontsize=8, color="#1B5E20")
 
    # ── Violations ──
    axes[2].plot(batches, viols, color="#F44336", linewidth=1.2, alpha=0.5, marker="o",
                 markersize=3, label="Avg violations")
    axes[2].plot(batches, _smooth(viols, 10), color="#B71C1C", linewidth=2.5, label="Smoothed")
    axes[2].plot(batches, _trend_line(batches, viols), color="#FF5722", linewidth=1.8,
                 linestyle="--", label="Trend")
    axes[2].set_ylabel("Deadline Violations", fontsize=11)
    axes[2].set_xlabel("Batch", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
 
    axes[2].annotate(f"{viols[0]:.0f}", (batches[0], viols[0]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color="#B71C1C")
    axes[2].annotate(f"{viols[-1]:.0f}", (batches[-1], viols[-1]),
                     textcoords="offset points", xytext=(-25, 5), fontsize=8, color="#B71C1C")
 
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "batch_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Batch summary saved → {path}")
    return path
 
 
# -----------------------------------------------------------------------------
# 3. Training summary — printed to console at end of training
# -----------------------------------------------------------------------------
 
def print_training_summary(batch_logs: List[Dict]):
    """
    Print a clean summary table to console after all batches complete.
    Call this from train.py after the training loop finishes.
    """
    if not batch_logs:
        return
 
    rewards = [b["avg_reward"]      for b in batch_logs]
    improv  = [b["avg_improvement"] for b in batch_logs]
    viols   = [b["avg_violations"]  for b in batch_logs]
    batches = [b["batch"]           for b in batch_logs]
 
    # First vs last 10 batch averages
    n = min(10, len(batch_logs))
    early_r = np.mean(rewards[:n])
    late_r  = np.mean(rewards[-n:])
    early_i = np.mean(improv[:n])
    late_i  = np.mean(improv[-n:])
    early_v = np.mean(viols[:n])
    late_v  = np.mean(viols[-n:])
 
    best_reward_batch = batches[int(np.argmax(rewards))]
    best_improv_batch = batches[int(np.argmax(improv))]
    best_viols_batch  = batches[int(np.argmin(viols))]
 
    print("\n" + "="*62)
    print("  PPO-ALNS TRAINING COMPLETE — SUMMARY")
    print("="*62)
    print(f"  Total batches trained : {len(batch_logs)}")
    print(f"  Total episodes        : {sum(b.get('episodes', 0) for b in batch_logs)}")
    print()
    print(f"  {'Metric':<28} {'First 10':>10} {'Last 10':>10} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Avg Reward':<28} {early_r:>10.3f} {late_r:>10.3f} {late_r - early_r:>+10.3f}")
    print(f"  {'Avg Improvement %':<28} {early_i:>10.2f} {late_i:>10.2f} {late_i - early_i:>+10.2f}")
    print(f"  {'Avg Violations':<28} {early_v:>10.1f} {late_v:>10.1f} {late_v - early_v:>+10.1f}")
    print()
    print(f"  Best reward     : {max(rewards):.4f}  (batch {best_reward_batch})")
    print(f"  Best improvement: {max(improv):.2f}%  (batch {best_improv_batch})")
    print(f"  Least violations: {min(viols):.1f}   (batch {best_viols_batch})")
    print()
 
    # Learning assessment
    reward_improved = late_r > early_r + 0.1
    improv_improved = late_i > early_i + 0.5
    viols_improved  = late_v < early_v - 5
 
    status = []
    if reward_improved:  status.append("Reward      ✓ IMPROVED")
    else:                status.append("Reward      ~ FLAT")
    if improv_improved:  status.append("Improvement ✓ IMPROVED")
    else:                status.append("Improvement ~ FLAT")
    if viols_improved:   status.append("Violations  ✓ REDUCED")
    else:                status.append("Violations  ~ FLAT")
 
    print("  Learning assessment:")
    for s in status:
        print(f"    {s}")
 
    print()
    print("  Next step: run demo comparison")
    print("    python src/main.py --seed 42 --start_time 09:00")
    print("    python src/main.py --use_ppo --seed 42 --start_time 09:00")
    print("="*62 + "\n")
 
 
# -----------------------------------------------------------------------------
# 4. Route map
# -----------------------------------------------------------------------------
 
def plot_route(instance, solution, title: str = ""):
    from constraints import RouteNode
 
    route      = solution.route
    orders_map = {o.order_id: o for o in instance.orders}
 
    depot_lat = instance.depot.lat
    depot_lon = instance.depot.lon
 
    pickup_lats, pickup_lons, pickup_labels       = [], [], []
    delivery_lats, delivery_lons, delivery_labels = [], [], []
    route_lats, route_lons = [depot_lat], [depot_lon]
 
    for rn in route:
        n = rn.node
        route_lats.append(n.lat)
        route_lons.append(n.lon)
        if rn.is_pickup:
            pickup_lats.append(n.lat)
            pickup_lons.append(n.lon)
            pickup_labels.append(rn.order_id[-3:])
        else:
            delivery_lats.append(n.lat)
            delivery_lons.append(n.lon)
            delivery_labels.append(rn.order_id[-3:])
 
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(route_lons, route_lats, color="#90CAF9", linewidth=0.8, alpha=0.6, zorder=1)
 
    for i in range(0, len(route_lons) - 1, 5):
        ax.annotate("", xy=(route_lons[i+1], route_lats[i+1]),
                    xytext=(route_lons[i], route_lats[i]),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8), zorder=2)
 
    ax.scatter(pickup_lons, pickup_lats, c="#2196F3", s=60, zorder=4,
               label="Pickup", marker="o", edgecolors="white", linewidths=0.5)
    ax.scatter(delivery_lons, delivery_lats, c="#F44336", s=60, zorder=4,
               label="Delivery", marker="s", edgecolors="white", linewidths=0.5)
    ax.scatter([depot_lon], [depot_lat], c="#4CAF50", s=200, zorder=5,
               label="Depot", marker="*", edgecolors="black", linewidths=0.8)
 
    for la, lo, lbl in zip(pickup_lats[:20], pickup_lons[:20], pickup_labels[:20]):
        ax.annotate(lbl, (lo, la), fontsize=5, ha="center", va="bottom",
                    color="#0D47A1", xytext=(0, 3), textcoords="offset points")
 
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title_str = title or f"Route — {instance.instance_id}"
    ax.set_title(f"{title_str}\n{len(route)} stops | {len(instance.orders)} orders | "
                 f"{instance.vehicle.vehicle_type} ({instance.vehicle.fuel_type})", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
 
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{instance.instance_id}_route_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Route map saved → {path}")
    return path
 
 
# -----------------------------------------------------------------------------
# 5. Cost breakdown pie chart
# -----------------------------------------------------------------------------
 
def plot_cost_breakdown(metrics: Dict, lateness: float, instance_id: str = ""):
    from alns_env import W_TRAVEL_TIME, W_LATENESS, W_CARBON, W_FUEL
 
    travel_cost = W_TRAVEL_TIME * metrics["travel_time_min"]
    late_cost   = W_LATENESS    * lateness
    carbon_cost = W_CARBON      * metrics["carbon_emission_kg"]
    fuel_cost   = W_FUEL        * metrics["fuel_cost_inr"]
 
    labels = ["Travel Time", "Lateness Penalty", "Carbon Emission", "Fuel Cost"]
    values = [travel_cost, late_cost, carbon_cost, fuel_cost]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
 
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return
    labels, values, colors = zip(*filtered)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(9)
 
    ax.set_title(f"Objective Cost Breakdown\n{instance_id}", fontsize=11)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{instance_id}_cost_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Cost breakdown saved → {path}")
    return path
 
 
# -----------------------------------------------------------------------------
# 6. Operator usage bar chart
# -----------------------------------------------------------------------------
 
def plot_operator_usage(destroy_counts: List[int], repair_counts: List[int],
                        instance_id: str = ""):
    destroy_names = ["Random", "Worst Cost", "Shaw", "String", "Geo Segment"]
    repair_names  = ["Greedy", "Criticality", "Regret-2"]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Operator Usage — {instance_id}", fontsize=12)
 
    ax1.bar(destroy_names, destroy_counts, color="#2196F3", edgecolor="white")
    ax1.set_title("Destroy Operators")
    ax1.set_ylabel("Times Selected")
    ax1.tick_params(axis="x", rotation=15)
    for i, v in enumerate(destroy_counts):
        ax1.text(i, v + 0.5, str(v), ha="center", fontsize=9)
 
    ax2.bar(repair_names, repair_counts, color="#4CAF50", edgecolor="white")
    ax2.set_title("Repair Operators")
    ax2.set_ylabel("Times Selected")
    for i, v in enumerate(repair_counts):
        ax2.text(i, v + 0.5, str(v), ha="center", fontsize=9)
 
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{instance_id}_operator_usage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Operator usage saved → {path}")
    return path