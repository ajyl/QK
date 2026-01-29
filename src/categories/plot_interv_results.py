import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pickle


def load_results(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_results(results, out_path: Path):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    heads = sorted(results.keys())
    n_heads = len(heads)
    if n_heads == 0:
        raise ValueError("No heads found in results.")

    ncols = min(4, n_heads)
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3.4 * ncols, 2.4 * nrows)
    )
    if n_heads == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    group_labels = ["Null Interv.", "Interv.\n(Rank = 5)", "Rand Baseline\n(Rank = 5)"]
    bar_labels = ["Orig Token", "Target Token"]
    colors = ["#4C78A8", "#F58518"]

    for idx, (ax, head) in enumerate(zip(axes, heads)):
        metrics = results[head]
        baseline = metrics["random_baseline"]
        values = [
            metrics["mean_orig_a"],
            metrics["mean_orig_a_target"],
            metrics["mean_interv_a"],
            metrics["mean_interv_a_target"],
            baseline["mean_interv_a"],
            baseline["mean_interv_a_target"],
        ]

        group_centers = range(len(group_labels))
        bar_width = 0.35
        offsets = [-bar_width / 2, bar_width / 2]

        ax.bar(
            [x + offsets[0] for x in group_centers],
            [values[0], values[2], values[4]],
            width=bar_width,
            color=colors[0],
            label=bar_labels[0],
        )
        ax.bar(
            [x + offsets[1] for x in group_centers],
            [values[1], values[3], values[5]],
            width=bar_width,
            color=colors[1],
            label=bar_labels[1],
        )
        ax.set_xticks(list(group_centers))
        ax.set_xticklabels(group_labels)
        ax.set_title(f"L{head[0]}H{head[1]}")
        ax.set_ylabel("Attention")
        ax.set_ylim(bottom=0.0)
        #if idx == n_heads - 1:
        if idx == 3:
            ax.legend(loc="upper right", bbox_to_anchor=(0.73, 1.0))

    for ax in axes[n_heads:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot intervention results.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to category_causal_intervention_results.pkl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to results path with .png",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out) if args.out else results_path.with_suffix(".png")

    results = load_results(results_path)
    plot_results(results, out_path)


if __name__ == "__main__":
    main()
