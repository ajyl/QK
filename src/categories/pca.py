import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from hook_utils import convert_to_hooked_model, seed_all
from dgp.entities import (
    LIQUIDS,
    COUNTRIES,
    VEHICLES,
    ANIMALS,
    FRUITS,
    SPORTS,
    MALE_NAMES,
    FEMALE_NAMES,
)

from categories.C_filter_heads import (
    build_C_filter_heads,
    index_query_key_vecs,
)
from categories.filter_head_utils import (
    build_filter_dataset,
    collect_item_timesteps_from_input_ids,
)
from util_funcs import get_qk_subspace, run_forward_pass
from constants import BASE_DIR, key_module_name, query_module_name

# %%

CATEGORIES = {
    "animal": ANIMALS,
    "fruit": FRUITS,
    "liquid": LIQUIDS,
    "vehicle": VEHICLES,
    "country": COUNTRIES,
}

# %%


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def build_item_category_map(categories):
    item2cat = {}
    for cat, words in categories.items():
        for w in words:
            item2cat[w] = cat
    return item2cat


def pos_qr(A):
    Q, R = torch.linalg.qr(A)
    signs = torch.diag(torch.sign(torch.diag(R)))
    Q = Q @ signs
    return Q


# %%

model_name = "meta-llama/Llama-3.1-8B-Instruct"
seed = 11
batch_size = 16

subspace_num_samples = 2000
n_per_category = 5
plot_num_samples = 200
task = "select_all"

subspace_thresh = 0.99
include_queries = True

heads = [(16, 19), (20, 26), (20, 14)]
output_dir = os.path.join(BASE_DIR, "figures/category_subspace_pca")

# %%

seed_all(seed)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
convert_to_hooked_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

hook_layers = sorted(list({layer for (layer, _) in heads}))
record_module_names = [
    key_module_name.format(hook_layer) for hook_layer in hook_layers
] + [query_module_name.format(hook_layer) for hook_layer in hook_layers]

categories = sorted(CATEGORIES.keys())
item2cat = build_item_category_map(CATEGORIES)
cat2idx = {c: i for i, c in enumerate(categories)}

# %%

subspace_samples = build_filter_dataset(
    CATEGORIES,
    n_samples=subspace_num_samples,
    n_per_category=n_per_category,
    task=task,
    seed=seed,
)
C_pos, C_neg, _, counts, _ = build_C_filter_heads(
    model,
    tokenizer,
    record_module_names,
    subspace_samples,
    heads,
    batch_size,
    categories=categories,
)

# %%

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups

# %%

plot_samples = build_filter_dataset(
    CATEGORIES,
    n_samples=plot_num_samples,
    n_per_category=n_per_category,
    task=task,
    seed=seed + 1,
)

key_vecs_per_head = {head: [] for head in heads}
query_vecs_per_head = {head: [] for head in heads}
labels_per_head = {head: [] for head in heads}
query_labels_per_head = {head: [] for head in heads}
for batch_start in range(0, len(plot_samples), batch_size):
    batch = plot_samples[batch_start : batch_start + batch_size]
    if not batch:
        continue

    prompts = [x["raw_input"] for x in batch]
    cache, _, input_ids, _ = run_forward_pass(
        model, tokenizer, record_module_names, prompts
    )

    item_pos = collect_item_timesteps_from_input_ids(
        tokenizer, input_ids, [x["items"] for x in batch], prefer_last_token=True
    )
    item_pos = torch.tensor(item_pos, device=model.device)

    labels = []
    query_labels = []
    for x in batch:
        row = []
        for item in x["items"]:
            if item not in item2cat:
                raise ValueError(f"Unknown item '{item}' not in CATEGORIES.")
            row.append(cat2idx[item2cat[item]])
        labels.append(row)
        query_labels.append(x["predicate"])
    labels = torch.tensor(labels, device=model.device)

    for layer_idx in hook_layers:
        q_vecs, k_vecs = index_query_key_vecs(
            cache,
            layer_idx,
            item_pos.to(model.device),
            num_key_value_groups,
        )
        layer_heads = [h for h in heads if h[0] == layer_idx]
        if not layer_heads:
            continue

        labels_flat = labels.reshape(-1).cpu().numpy()
        for head in layer_heads:
            _, head_idx = head
            k_head = k_vecs[:, head_idx, :, :].reshape(-1, k_vecs.shape[-1])
            q_head = q_vecs[:, head_idx]  # [B, d_head]
            key_vecs_per_head[head].append(k_head)
            query_vecs_per_head[head].append(q_head)
            labels_per_head[head].append(labels_flat)
            query_labels_per_head[head].append(
                np.array([cat2idx[x] for x in query_labels])
            )

# %%

categories = [cat[0].upper() + cat[1:] for cat in categories]

for head in [(20, 26)]:
    head_dir = output_dir / f"L{head[0]}_H{head[1]}"
    head_dir.mkdir(parents=True, exist_ok=True)

    if len(key_vecs_per_head[head]) == 0:
        print(f"Head {head}: no key vectors collected, skipping.")
        continue

    total = int(counts[head].sum().item())
    if total == 0:
        print(f"Head {head}: no counts for subspace, skipping.")
        continue

    _keys = []
    _queries = []
    for c_idx in range(len(categories)):
        U, S, Vh, rank = get_qk_subspace(
            C_pos[head][c_idx],
            C_neg[head][c_idx],
            thresh=subspace_thresh,
        )
        _keys.append(Vh[:rank, :])
        _queries.append(U[:, :rank])

    _keys = torch.cat(_keys, dim=0)  # [sum_ranks, d_head]
    _queries = torch.cat(_queries, dim=1)  # [d_head, sum_ranks]
    key_basis = pos_qr(_keys.T)
    query_basis = pos_qr(_queries)

    rank = key_basis.shape[1]
    assert rank == query_basis.shape[1]
    if rank < 3:
        print(f"Head {head}: rank={rank}, need >=3 for 3D PCA, skipping.")
        continue

    k_all = torch.cat(key_vecs_per_head[head], dim=0)
    q_all = torch.cat(query_vecs_per_head[head], dim=0)
    key_labels_all_idx = np.concatenate(labels_per_head[head], axis=0)
    query_labels_all_idx = np.concatenate(query_labels_per_head[head], axis=0)

    k_coords = k_all @ key_basis  # [N, rank]
    q_coords = q_all @ query_basis
    pca = PCA(n_components=3)
    if include_queries:
        combined_coords = torch.cat([k_coords, q_coords], dim=0)
        pca_result = pca.fit_transform(combined_coords.numpy())
        data_type = ["Key"] * k_coords.shape[0] + ["Query"] * q_coords.shape[0]
        labels_all = np.array(
            [
                categories[idx]
                for idx in np.concatenate(
                    [key_labels_all_idx, query_labels_all_idx], axis=0
                )
            ]
        )

    else:
        pca_result = pca.fit_transform(k_coords.numpy())
        data_type = ["Key"] * k_coords.shape[0]
        labels_all = np.array([categories[idx] for idx in key_labels_all_idx])

    key_marker_size = 0.75
    query_marker_size = 4

    df = pd.DataFrame(
        {
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "PCA3": pca_result[:, 2],
            "Category": labels_all,
            "DataType": data_type,
        }
    )
    fig = px.scatter_3d(
        df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Category",
        title=f"Category subspace<br>(Layer {head[0]}, Head {head[1]})",
        symbol="DataType" if include_queries else None,
        symbol_map={"Key": "circle", "Query": "cross"},
        labels={
            "DataType": "Query vs. Key",
            # "Category": "",
        },
    )
    fig.update_layout(legend_title_text="Query vs. Key")
    # update marker sizes manually
    for i, trace in enumerate(fig.data):
        marker_size = key_marker_size
        if include_queries and trace.marker.symbol == "cross":
            marker_size = query_marker_size
        fig.data[i].marker.size = marker_size
        trace.showlegend = False

    fig.update_traces(marker={"opacity": 1.0})

    category_colors = {}
    for trace in fig.data:
        category = trace.name.split(",")[0]
        if category not in category_colors:
            category_colors[category] = trace.marker.color

    x = 0.4 * 1.4
    y = 0.75
    z = 0.5
    scale = 2
    scene_settings = dict(
        camera=dict(
            eye=dict(
                x=x * scale,
                y=y * scale,
                z=z * scale,
            )
        )
    )

    scene_settings["aspectmode"] = "manual"
    scene_settings["aspectratio"] = dict(x=0.8, y=1, z=1)

    fig.update_layout(scene=scene_settings)
    legend_items = [
        ("Key", "black", "circle"),
        ("Query", "black", "cross"),
    ] + [(category, color, "circle") for category, color in category_colors.items()]

    if legend_items:
        legend_x = 0.92
        legend_y = 0.5
        line_height = 0.06
        pad = 0.08
        box_width = 0.28
        box_height = line_height * len(legend_items) + pad * 2
        box_x0 = legend_x - box_width / 2
        box_x1 = legend_x + box_width / 2
        box_y0 = legend_y - box_height / 2
        box_y1 = legend_y + box_height / 2

        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=box_x0,
            x1=box_x1,
            y0=box_y0,
            y1=box_y1,
            line=dict(color="rgba(0,0,0,0.2)", width=1),
            fillcolor="rgba(255,255,255,0.9)",
            layer="above",
        )

        annotations = []
        for i, (label, color, marker) in enumerate(legend_items):
            marker_text = "●" if marker == "circle" else "+"
            y_pos = box_y1 - pad - i * line_height
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=box_x0 + 0.02,
                    y=y_pos,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    text=marker_text,
                    font=dict(size=9, color=color),
                )
            )
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=box_x0 + 0.065,
                    y=y_pos,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    text=label,
                    font=dict(size=8, color="black"),
                )
            )
        fig.update_layout(annotations=annotations, showlegend=False)
    else:
        fig.update_layout(showlegend=False)
    fig.update_layout(
        font=dict(family="Times New Roman, Times, serif", size=9),
        margin=dict(l=40, r=0, t=10, b=40),
        width=240,
        height=210,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_scenes(
        xaxis_showticklabels=False,
        xaxis_ticks="",
        xaxis_title_text="",
        yaxis_showticklabels=False,
        yaxis_ticks="",
        yaxis_title_text="",
        zaxis_showticklabels=False,
        zaxis_ticks="",
        zaxis_title_text="",
        xaxis=dict(
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.90)",
        ),
        yaxis=dict(
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.90)",
        ),
        zaxis=dict(
            showbackground=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.90)",
        ),
        bgcolor="white",
    )
    fig.add_annotation(
        x=0.84, y=0.07, xref="paper", yref="paper",
        text="PC 1", showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        x=0.17, y=0.12, xref="paper", yref="paper",
        text="PC 2", showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        x=0.02, y=0.57, xref="paper", yref="paper",
        text="PC 3", showarrow=False, font=dict(size=9)
    )
    fig.update_layout(
        title={
            "x": 0.55,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "top",
            "font": dict(size=10),
        }
    )
    suffix = "_with_queries" if include_queries else ""
    out_name = sanitize_name(f"category_subspace_pca_global{suffix}") + ".html"
    # Save to pdf
    fig.write_image(head_dir / out_name.replace(".html", ".pdf"))
    fig.write_html(head_dir / out_name)


print("Done.")

# %%

