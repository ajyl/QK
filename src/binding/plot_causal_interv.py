import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from constants import BASE_DIR


with open("causal_intervention_results_9_entities.pkl", "rb") as f:
    metrics = pickle.load(f)

filtered_heads = [
    (16, 1),
    (16, 19),
    (17, 24),
]
num_heads = len(filtered_heads)
num_cols = 3
num_rows = (num_heads + num_cols - 1) // num_cols
fig, axes = plt.subplots(
    nrows=num_rows, ncols=num_cols, figsize=(3.8 * num_cols, 2.4 * num_rows)
)

axes = np.array(axes).reshape(-1)
subspaces = ["order", "lexical", "both"]
subspace_labels = {
    "order": "Order Subspace",
    "lexical": "Lexical Subspace",
    "both": "Both Subspaces",
    "order_baseline": "Random Baseline",
    "lexical_baseline": "Random Baseline",
    "both_baseline": "Random Baseline",
}

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
for i, head in enumerate(filtered_heads):
    ax = axes[i]
    base_metrics = metrics[(head, "order")]
    order_metrics = metrics[(head, "order")]
    lexical_metrics = metrics[(head, "lexical")]
    both_metrics = metrics[(head, "both")]

    base_answer = base_metrics["mean_orig_a"]
    base_target = base_metrics["mean_orig_a_target"]

    order_answer = order_metrics["mean_interv_a"]
    order_target = order_metrics["mean_interv_a_target"]
    lexical_answer = lexical_metrics["mean_interv_a"]
    lexical_target = lexical_metrics["mean_interv_a_target"]
    both_answer = both_metrics["mean_interv_a"]
    both_target = both_metrics["mean_interv_a_target"]

    order_baseline = metrics[(head, "order_baseline")]
    order_baseline_orig = order_baseline["mean_orig_a"]
    order_baseline_target = order_baseline["mean_orig_a_target"]

    lexical_baseline = metrics[(head, "lexical_baseline")]
    lexical_baseline_orig = lexical_baseline["mean_orig_a"]
    lexical_baseline_target = lexical_baseline["mean_orig_a_target"]

    both_baseline = metrics[(head, "both_baseline")]
    both_baseline_orig = both_baseline["mean_orig_a"]
    both_baseline_target = both_baseline["mean_orig_a_target"]

    width = 0.35

    answers = [
        base_answer,
        order_answer,
        lexical_answer,
        both_answer,
        order_baseline_orig,
        lexical_baseline_orig,
        both_baseline_orig,
    ]
    targets = [
        base_target,
        order_target,
        lexical_target,
        both_target,
        order_baseline_target,
        lexical_baseline_target,
        both_baseline_target,
    ]
    x = np.arange(len(answers))
    ax.bar(
        x - width / 2,
        answers,
        width,
        label="Orig Token",
        # alpha=0.8,
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        targets,
        width,
        label="Target Token",
        alpha=0.8,
        color="#F58518",
    )

    # Set y limit:
    # ax.set_ylim(0, 1.02)
    ax.set_title(f"Layer {head[0]} Head {head[1]}")
    ax.set_xticks(x)

    order_rank = order_metrics["rank"]
    lex_rank = lexical_metrics["rank"]
    ax.set_xticklabels(
        [
            "Null\nInterv.",
            f"Order\n(r: {order_rank})",
            f"Lex.\n(r: {lex_rank})",
            f"Both\n(r: {order_rank + lex_rank})",
            f"Rand.\n(r: {order_rank})",
            f"Rand.\n(r: {lex_rank})",
            f"Rand.\n(r: {order_rank + lex_rank})",
        ],
        fontsize=9,
    )
    if i % num_cols == 0:
        ax.set_ylabel("Mean Attention")
    if i == 2:
        ax.legend(fontsize=8, loc="upper right")

    ax.axvline(0.5, linestyle="--", color="0.5", linewidth=0.8)
    ax.axvline(3.5, linestyle="--", color="0.5", linewidth=0.8)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"figures/binding_causal_interv_llama.pdf"))
