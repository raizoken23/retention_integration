# python plot.py --generate_stats generate_stats.json --output generation_time.png --extend 100000
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot generation timing statistics")
    parser.add_argument(
        "--generate_stats",
        type=str,
        required=True,
        help="Path to generate_stats.json produced by generate.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save figure (e.g., plot.png/pdf). If omitted, shows interactively.",
    )
    parser.add_argument(
        "--duration-plot",
        action="store_true",
        default=False,
        help="Plot duration instead of throughput.",
    )
    parser.add_argument(
        "--extend",
        type=int,
        default=None,
        help="If provided, fit a polynomial to cumulative time and extend to this many tokens (dashed).",
    )
    return parser.parse_args()


def load_instances(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    with p.open("r") as f:
        content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        # Support single-record file just in case
        return [data]


def reconstruct_token_durations(instance: Dict[str, Any]) -> List[float]:
    """
    Reconstruct per-token durations (seconds) from the stats record.

    generate.py collects per-token throughput for streamed tokens as tokens_per_second
    (excluding first and last token). We reconstruct durations by:
      - first token duration: time_to_first_token
      - middle tokens: duration_i = 1 / tokens_per_second[i]
      - last token duration: residual = total_time - sum(durations_so_far)
    """
    time_to_first = float(instance.get("time_to_first_token", 0.0) or 0.0)
    tokens_per_second = list(instance.get("tokens_per_second", []))
    total_time = float(instance.get("total_time", 0.0) or 0.0)

    durations: List[float] = []
    if time_to_first > 0:
        durations.append(time_to_first)
    # middle tokens
    for tps in tokens_per_second:
        if tps and tps > 0:
            durations.append(1.0 / float(tps))
    # last token residual (can be zero or tiny negative due to float; clamp at >=0)
    residual = max(0.0, total_time - sum(durations))
    if residual > 0:
        durations.append(residual)
    # Drop first and last elements which are typically less reliable
    if len(durations) >= 3:
        return durations[1:-1]
    return []


def label_for_instance(instance: Dict[str, Any]) -> str:
    args = instance.get("args", {}) or {}
    model = instance.get("model", {}) or {}
    tokenizer = instance.get("tokenizer", {}) or {}
    label = instance.get("label", "") or ""

    if label:
        return label

    parts = []
    if model.get("name_or_path"):
        parts.append(str(model["name_or_path"]))
    if args.get("chunk_size") is not None:
        parts.append(f"chunk={args.get('chunk_size')}")
    if args.get("switch_over_seq_len") is not None:
        parts.append(f"switch={args.get('switch_over_seq_len')}")
    if args.get("max_new_tokens") is not None:
        parts.append(f"max_new={args.get('max_new_tokens')}")
    if tokenizer.get("name_or_path"):
        parts.append(str(tokenizer["name_or_path"]))
    return " | ".join(parts) if parts else "run"


def main() -> None:
    args = parse_args()
    instances = load_instances(args.generate_stats)
    if not instances:
        raise SystemExit("No records found in stats file.")

    # Prepare figure with two vertically stacked subplots
    plt.style.use("seaborn-v0_8-whitegrid")
    if args.duration_plot:
        fig, (ax_cum, ax_per) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    else:
        fig, (ax_cum) = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    for idx, inst in enumerate(instances):
        durations = reconstruct_token_durations(inst)
        if not durations:
            continue
        # x-axis: token indices starting at 1
        token_idx = list(range(1, len(durations) + 1))
        # cumulative times (seconds)
        cum = []
        s = 0.0
        for d in durations:
            s += d
            cum.append(s)

        label = label_for_instance(inst)

        # Plot cumulative
        cum_minutes = [v / 60.0 for v in cum]
        line_cum = ax_cum.plot(token_idx, cum_minutes, label=label, linewidth=1.8)[0]
        # Plot per-token duration
        if args.duration_plot:
            ax_per.plot(token_idx, durations, label=label, linewidth=1.2, alpha=0.9)

        # Optional extension via polynomial fit (quadratic by default)
        if args.extend is not None and isinstance(args.extend, int) and args.extend > len(durations):
            x_fit = np.array(token_idx, dtype=float)
            y_fit = np.array(cum, dtype=float)
            # Degree-2 polynomial fit for smooth curvature
            coeffs = np.polyfit(x_fit, y_fit, deg=2)
            x_ext = np.arange(len(durations) + 1, args.extend + 1, dtype=float)
            if x_ext.size > 0:
                y_ext_seconds = np.polyval(coeffs, x_ext)
                y_ext_minutes = (y_ext_seconds / 60.0).tolist()
                ax_cum.plot(
                    x_ext.tolist(),
                    y_ext_minutes,
                    linestyle="--",
                    linewidth=1.6,
                    color=line_cum.get_color(),
                )
                # annotate end with simulated total runtime in hours
                total_hours = float(y_ext_seconds[-1]) / 3600.0
                if total_hours < 1.:
                    total_time = total_hours * 60.0
                else:
                    total_time = total_hours
                ax_cum.annotate(
                    f"{total_time:.2f} {'min' if total_hours < 1. else 'h'}",
                    xy=(x_ext[-1], y_ext_minutes[-1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=line_cum.get_color(),
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

    # Styling cumulative plot
    ax_cum.set_title("Cumulative token generation time")
    ax_cum.set_xlabel("Number of tokens")
    ax_cum.set_ylabel("Cumulative time (min)")
    ax_cum.legend(frameon=False)

    # Styling per-token duration plot
    if args.duration_plot:
        ax_per.set_title("Per-token generation time")
        ax_per.set_xlabel("Number of tokens")
        ax_per.set_ylabel("Time per token (s)")
        ax_per.set_yscale("log")
        ax_per.legend(frameon=False)

    # Tight aesthetic for publication-ready look
    for ax in (ax_cum, ax_per) if args.duration_plot else (ax_cum,):
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)

    if args.output:
        out = Path(args.output)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()


