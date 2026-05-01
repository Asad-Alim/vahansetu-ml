"""
output.py
---------
Processes 4 benchmark files and saves separate graph images for each.

Files:
  1. 2nd_benchmark.txt
  2. benchmark_result.txt
  3. 1st_benchmark.txt
  4. benchmark_29.txt
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_run_lines(text: str) -> list:
    pattern = re.compile(
        r"\[\s*\d+/\d+\]\s+"
        r"(INST\d+_N(\d+)_\w+)"
        r"\s+run=\s*(\d+)"
        r"\s+greedy_viol=\s*(\d+)"
        r"\s+ppo_viol=\s*(\d+)"
        r"\s+greedy_cost=\s*([\d.]+)"
        r"\s+ppo_cost=\s*([\d.]+)"
    )
    records = []
    seen = set()
    for m in pattern.finditer(text):
        key = (m.group(1), m.group(3))
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "instance":    m.group(1),
            "order_size":  int(m.group(2)),
            "run":         int(m.group(3)),
            "greedy_viol": int(m.group(4)),
            "ppo_viol":    int(m.group(5)),
            "greedy_cost": float(m.group(6)),
            "ppo_cost":    float(m.group(7)),
        })
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(records: list) -> dict:
    buckets = defaultdict(lambda: {
        "ppo_cost": [], "greedy_cost": [],
        "ppo_viol": [], "greedy_viol": []
    })
    for r in records:
        n = r["order_size"]
        buckets[n]["ppo_cost"].append(r["ppo_cost"])
        buckets[n]["greedy_cost"].append(r["greedy_cost"])
        buckets[n]["ppo_viol"].append(r["ppo_viol"])
        buckets[n]["greedy_viol"].append(r["greedy_viol"])

    result = {}
    for n in sorted(buckets):
        b = buckets[n]
        result[n] = {
            "ppo_cost_mean":    np.mean(b["ppo_cost"]),
            "ppo_cost_std":     np.std(b["ppo_cost"]),
            "greedy_cost_mean": np.mean(b["greedy_cost"]),
            "greedy_cost_std":  np.std(b["greedy_cost"]),
            "ppo_viol_mean":    np.mean(b["ppo_viol"]),
            "ppo_viol_std":     np.std(b["ppo_viol"]),
            "greedy_viol_mean": np.mean(b["greedy_viol"]),
            "greedy_viol_std":  np.std(b["greedy_viol"]),
            "n_samples":        len(b["ppo_cost"]),
        }
    return result


# ── Plot 1 : Cost vs Order Size ───────────────────────────────────────────────

def plot_cost(agg: dict, out: str, title_prefix: str):
    sizes        = sorted(agg.keys())
    ppo_means    = [agg[n]["ppo_cost_mean"]    for n in sizes]
    greedy_means = [agg[n]["greedy_cost_mean"] for n in sizes]
    ppo_stds     = [agg[n]["ppo_cost_std"]     for n in sizes]
    greedy_stds  = [agg[n]["greedy_cost_std"]  for n in sizes]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(sizes, ppo_means,    marker="o", linewidth=2.2, color="#2563EB", label="PPO")
    ax.fill_between(sizes,
                    [m - s for m, s in zip(ppo_means, ppo_stds)],
                    [m + s for m, s in zip(ppo_means, ppo_stds)],
                    alpha=0.15, color="#2563EB")

    ax.plot(sizes, greedy_means, marker="s", linewidth=2.2, color="#DC2626", label="Greedy")
    ax.fill_between(sizes,
                    [m - s for m, s in zip(greedy_means, greedy_stds)],
                    [m + s for m, s in zip(greedy_means, greedy_stds)],
                    alpha=0.15, color="#DC2626")

    for x, ym, yg in zip(sizes, ppo_means, greedy_means):
        ax.annotate(f"{ym:.0f}", (x, ym), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#2563EB")
        ax.annotate(f"{yg:.0f}", (x, yg), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color="#DC2626")

    ax.set_xticks(sizes)
    ax.set_xlabel("Order Size (N)", fontsize=13)
    ax.set_ylabel("Average Cost", fontsize=13)
    ax.set_title(f"[{title_prefix}] PPO vs Greedy — Average Cost by Order Size\n(shaded = ±1 std dev)", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":",  alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  [✓] Saved → {out}")
    plt.show()


# ── Plot 2 : Cost Gap ─────────────────────────────────────────────────────────

def plot_cost_gap(agg: dict, out: str, title_prefix: str):
    sizes  = sorted(agg.keys())
    gaps   = [agg[n]["ppo_cost_mean"] - agg[n]["greedy_cost_mean"] for n in sizes]
    colors = ["#16A34A" if g < 0 else "#DC2626" for g in gaps]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(sizes, gaps, color=colors, width=0.7, edgecolor="white", linewidth=0.8)

    for bar, gap in zip(bars, gaps):
        va = "bottom" if gap >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                gap + (10 if gap >= 0 else -10),
                f"{gap:+.1f}", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(sizes)
    ax.set_xlabel("Order Size (N)", fontsize=13)
    ax.set_ylabel("PPO cost − Greedy cost", fontsize=13)
    ax.set_title(f"[{title_prefix}] Cost Gap: PPO − Greedy by Order Size\n(green = PPO better, red = Greedy better)", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  [✓] Saved → {out}")
    plt.show()


# ── Plot 3 : Deadline Violations ─────────────────────────────────────────────

def plot_violations(agg: dict, out: str, title_prefix: str):
    sizes        = sorted(agg.keys())
    ppo_means    = [agg[n]["ppo_viol_mean"]    for n in sizes]
    greedy_means = [agg[n]["greedy_viol_mean"] for n in sizes]
    ppo_stds     = [agg[n]["ppo_viol_std"]     for n in sizes]
    greedy_stds  = [agg[n]["greedy_viol_std"]  for n in sizes]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(sizes, ppo_means,    marker="o", linewidth=2.2, color="#7C3AED", label="PPO Deadline Violations")
    ax.fill_between(sizes,
                    [max(0, m - s) for m, s in zip(ppo_means, ppo_stds)],
                    [m + s for m, s in zip(ppo_means, ppo_stds)],
                    alpha=0.15, color="#7C3AED")

    ax.plot(sizes, greedy_means, marker="s", linewidth=2.2, color="#EA580C", label="Greedy Deadline Violations")
    ax.fill_between(sizes,
                    [max(0, m - s) for m, s in zip(greedy_means, greedy_stds)],
                    [m + s for m, s in zip(greedy_means, greedy_stds)],
                    alpha=0.15, color="#EA580C")

    for x, ym, yg in zip(sizes, ppo_means, greedy_means):
        ax.annotate(f"{ym:.2f}", (x, ym), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color="#7C3AED")
        ax.annotate(f"{yg:.2f}", (x, yg), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color="#EA580C")

    ax.set_xticks(sizes)
    ax.set_xlabel("Order Size (N)", fontsize=13)
    ax.set_ylabel("Average Deadline Violations", fontsize=13)
    ax.set_title(f"[{title_prefix}] PPO vs Greedy — Deadline Violations by Order Size\n(shaded = ±1 std dev)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":",  alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  [✓] Saved → {out}")
    plt.show()


# ── Process one file ──────────────────────────────────────────────────────────

def process_file(txt_path: str):
    p = Path(txt_path)
    if not p.exists():
        print(f"\n  [✗] File not found: {txt_path} — skipping")
        return

    print(f"\n{'='*60}")
    print(f"  Processing: {p.name}")
    print(f"{'='*60}")

    text    = p.read_text(encoding="utf-8", errors="replace")
    records = parse_run_lines(text)

    if not records:
        print(f"  [✗] No records found in {p.name} — skipping")
        return

    print(f"  [i] {len(records)} run records loaded")
    agg    = aggregate(records)
    prefix = p.stem                        # e.g. "2nd_benchmark"
    outdir = Path(f"graphs_{prefix}")      # separate folder per file
    outdir.mkdir(exist_ok=True)

    print(f"  [i] Order sizes: {sorted(agg.keys())}")
    print(f"  [i] Saving graphs to folder → {outdir}/")

    plot_cost(      agg, str(outdir / "cost_vs_order_size.png"),       prefix)
    plot_cost_gap(  agg, str(outdir / "cost_gap_vs_order_size.png"),   prefix)
    plot_violations(agg, str(outdir / "violations_vs_order_size.png"), prefix)


# ── ✏️ Add your 4 filenames here ─────────────────────────────────────────────

FILES = [
    "2nd_benchmark.txt",
    "benchmark_result.txt",
    "1st_benchmark.txt",
    "benchmark_29.txt",
]

for f in FILES:
    process_file(f)

print("\n[✓] All done!")