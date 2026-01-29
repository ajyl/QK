import os
from functools import partial
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from hook_utils import (
    convert_to_hooked_model,
    seed_all,
)

from dgp.build_data import build_prompt
from dgp.schemas import SCHEMA_BOXES
from dgp.dataset import BindingDataset
import binding.C_order as C_order
import binding.C_lex as C_lex
from util_funcs import get_qk_subspace
from constants import BASE_DIR, key_module_name, query_module_name

# %%

seed_all(11)

schema = SCHEMA_BOXES
num_instances = 9
max_objects = 9


objects = schema.items["Object"][:max_objects]
boxes = schema.items["Box"][:max_objects]
unused_objects = schema.items["Object"][max_objects:]
unused_boxes = schema.items["Box"][max_objects:]
schema.items["Object"] = objects
schema.items["Box"] = boxes
schema.unused_items["Object"] = unused_objects
schema.unused_items["Box"] = unused_boxes

# Objects
query_cat_id = 0
# Boxes
answer_cat_id = 1
order_tag_sampler = partial(
    build_prompt,
    schema=schema,
    num_instances=num_instances,
    query_cat_idx=query_cat_id,
    answer_cat_idx=answer_cat_id,
    query_same_item=True,
    query_from_unused=True,
)
lexical_tag_sampler = partial(
    build_prompt,
    schema=schema,
    num_instances=num_instances,
    query_cat_idx=query_cat_id,
    answer_cat_idx=answer_cat_id,
    query_same_item=False,
)
order_dataset = BindingDataset.from_sampler(order_tag_sampler, 3000)
lexical_dataset = BindingDataset.from_sampler(lexical_tag_sampler, 3000)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
convert_to_hooked_model(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

filtered_heads = [
    (16, 21),
    (19, 0),
]
hook_layers = set([layer for (layer, head) in filtered_heads])
record_module_names = [
    key_module_name.format(hook_layer) for hook_layer in hook_layers
] + [query_module_name.format(hook_layer) for hook_layer in hook_layers]
batch_size = 16
key_matcher = schema.matchers[answer_cat_id]
query_matcher = schema.matchers[query_cat_id]
num_heads = model.config.num_attention_heads
d_head = model.config.head_dim

# %%

(
    C_order_pos,
    C_order_neg,
    query_vecs_order,
    key_vecs_order,
    all_order_labels,
    q_labels,
) = C_order.build_C_order(
    model,
    tokenizer,
    record_module_names,
    order_dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
    num_instances,
)

# %%

(
    C_lex_pos,
    C_lex_neg,
    query_vecs_lex_pos,
    query_vecs_lex_neg,
    key_vecs_lex_pos,
    key_vecs_lex_neg,
    all_lexical_labels,
) = C_lex.build_C_lexical(
    model,
    tokenizer,
    record_module_names,
    lexical_dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
)

# %%


def make_discrete_colorscale(unique_labels, palette):
    """
    Build a stepwise colorscale so each label gets a solid block
    in the colorbar instead of a gradient.
    """
    unique_labels = sorted(unique_labels)
    n = len(unique_labels)
    if n == 0:
        return [[0.0, "black"], [1.0, "black"]]
    if n == 1:
        return [[0.0, palette[0]], [1.0, palette[0]]]

    scale = []
    for i, lab in enumerate(unique_labels):
        # evenly spaced bins in [0, 1]
        start = i / n
        end = (i + 1) / n
        col = palette[i % len(palette)]
        scale.append([start, col])
        scale.append([end, col])
    return scale


def plot_pca_subplot(
    C_pos,
    C_neg,
    query_vecs,
    key_vecs,
    labels,
    q_labels,
    discrete=False,
    keys_only=False,
):

    _c_pos = C_pos
    _c_neg = C_neg
    k_vecs = key_vecs
    q_vecs = query_vecs
    U, S, Vh, rank = get_qk_subspace(_c_pos, _c_neg, thresh=0.99)

    rank = 3
    U_tag = U[:, :rank]
    V_tag = Vh.T[:, :rank]

    q_scale = 2
    q_coords = q_vecs @ U_tag
    k_coords = k_vecs @ V_tag

    if keys_only:
        combined_coords = k_coords.numpy()
        combined_labels = labels
        combined_labels = combined_labels + 1
        combined_types = np.full(len(k_coords), "key")
    else:
        combined_coords = np.concatenate([k_coords.numpy(), q_coords.numpy()], axis=0)
        combined_labels = np.concatenate([labels, q_labels], axis=0)
        combined_labels = combined_labels + 1
        combined_types = np.concatenate(
            [
                np.full(len(k_coords), "key"),
                np.full(len(q_coords), "query"),
            ]
        )

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(combined_coords)
    df = pd.DataFrame(
        {
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "PCA3": pca_result[:, 2],
            "Label": combined_labels,
            "Type": combined_types,
        }
    )

    if discrete:
        unique_labels = sorted(df["Label"].unique())
        palette = px.colors.qualitative.D3
        continuous_scale = make_discrete_colorscale(unique_labels, palette)

        n_unique = max(len(unique_labels), 1)
        colorbar = dict(
            title="Lexical ID",
            tickvals=unique_labels,
            ticktext=[str(int(lab)) for lab in unique_labels],
            len=0.7,
            tickformat=".0f",
            ticks="outside",
        )
        color_field = df["Label"]
        # continuous_scale = palette

    else:
        colorbar = dict(
            title="Order ID",
            len=0.7,
            tickformat=".0f",
            ticks="outside",
        )
        color_field = df["Label"]
        continuous_scale = "Viridis"

    fig = px.scatter_3d(
        df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color=color_field,
        symbol="Type",
        color_continuous_scale=continuous_scale,
        symbol_map={"key": "circle", "query": "cross"},
        # title=f"Layer {_head[0]} Head {_head[1]}",
        labels={
            "Label": "Order ID",
            "Type": "Query vs. Key",
        },
    )
    return {"fig": fig, "colorbar": colorbar, "continuous_scale": continuous_scale}


def _label_color_and_symbols(labels, types):
    palette = px.colors.qualitative.D3
    label_strings = [str(lab) for lab in labels]
    unique_labels = sorted(set(label_strings))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}
    colors = [color_map[lab] for lab in label_strings]
    symbol_map = {"key": "circle", "query": "cross"}
    symbols = [symbol_map.get(t, "circle") for t in types]
    return colors, symbols


def plot_umap_subplot(
    C_pos,
    C_neg,
    q_vecs,
    k_vecs,
    labels,
    umap_args,
):
    U, S, Vh, rank = get_qk_subspace(C_pos, C_neg, thresh=0.99)

    U_tag = U[:, :rank]
    V_tag = Vh.T[:, :rank]

    q_coords = (q_vecs @ U_tag).cpu().numpy()
    k_coords = (k_vecs @ V_tag).cpu().numpy()
    label_values = labels.cpu().numpy()

    combined_coords = np.concatenate([k_coords, q_coords], axis=0)
    combined_labels = np.concatenate([label_values, label_values])
    combined_labels = combined_labels + 1
    combined_types = np.concatenate(
        [
            np.full(len(k_coords), "key"),
            np.full(len(q_coords), "query"),
        ]
    )

    umap_n_components = umap_args["n_components"]
    umap_neighbors = umap_args["n_neighbors"]
    umap_min_dist = umap_args["min_dist"]
    umap_metric = umap_args["metric"]
    umap_plot_scale = umap_args["plot_scale"]
    umap_seed = umap_args["random_state"]
    reducer = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_seed,
    )
    embedding = reducer.fit_transform(combined_coords)
    embedding = embedding * umap_plot_scale

    colors, symbols = _label_color_and_symbols(combined_labels, combined_types)
    df = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "UMAP3": embedding[:, 2] if umap_n_components == 3 else None,
            "Label": combined_labels,
            "Type": combined_types,
        }
    )
    if umap_n_components == 2:
        breakpoint()

    elif umap_n_components == 3:
        fig = px.scatter_3d(
            df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color=df["Label"],
            symbol="Type",
            color_continuous_scale=px.colors.qualitative.D3,
            symbol_map={"key": "circle", "query": "cross"},
            # labels={
            #    "Label": "Order ID",
            #    "Type": "Query vs. Key",
            # },
        )

    unique_labels = sorted(df["Label"].unique())
    palette = px.colors.qualitative.D3
    continuous_scale = make_discrete_colorscale(unique_labels, palette)
    n_unique = max(len(unique_labels), 1)
    colorbar = dict(
        title="Lexical ID",
        tickvals=unique_labels,
        ticktext=[str(int(lab)) for lab in unique_labels],
        len=0.7,
        tickformat=".0f",
        ticks="outside",
    )
    return {"fig": fig, "colorbar": colorbar, "continuous_scale": continuous_scale}


def make_plots(
    first,
    second,
    third,
):

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        subplot_titles=[
            "(a) Order-ID Subspace (PCA)",
            "(b) Lexical Subspace (PCA)",
            "(c) Lexical Subspace (UMAP)",
        ],
    )
    first_fig = plot_pca_subplot(
        first["C_pos"],
        first["C_neg"],
        first["q_vecs"],
        first["k_vecs"],
        first["labels"],
        first["q_labels"],
        discrete=False,
        keys_only=first.get("keys_only", False),
    )
    for trace in first_fig["fig"].data:
        fig.add_trace(trace, row=1, col=1)

    second_fig = plot_pca_subplot(
        second["C_pos"],
        second["C_neg"],
        second["q_vecs"],
        second["k_vecs"],
        second["labels"],
        second["q_labels"],
        discrete=True,
        keys_only=second.get("keys_only", False),
    )
    for trace in second_fig["fig"].data:
        fig.add_trace(trace, row=1, col=2)

    third_fig = plot_umap_subplot(
        third["C_pos"],
        third["C_neg"],
        third["q_vecs"],
        third["k_vecs"],
        third["labels"],
        third["umap_args"],
    )
    for trace in third_fig["fig"].data:
        fig.add_trace(trace, row=1, col=3)

    return (
        fig,
        first_fig,
        second_fig,
        third_fig,
    )


def _make_pretty(subfig, config):

    _subfig = subfig["fig"]
    colorbar = subfig["colorbar"]
    continuous_scale = subfig["continuous_scale"]

    camera = config["camera"]
    scale = camera.get("scale", 1.2)
    x = camera.get("x", -1.5)
    y = camera.get("y", -0.6)
    z = camera.get("z", 0.5)

    scene_settings = dict(
        camera=dict(
            eye=dict(
                x=x * scale,
                y=y * scale,
                z=z * scale,
            )
        )
    )
    aspectmode = config.get("aspectmode")
    aspectratio = config.get("aspectratio")
    if aspectmode:
        scene_settings["aspectmode"] = aspectmode
    if aspectratio:
        scene_settings["aspectratio"] = aspectratio

    _subfig.update_layout(
        scene=scene_settings,
        legend=dict(
            x=0.3,
            y=0.15,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0,0,0,0.2)",
        ),
        margin=dict(l=0, r=5, t=5, b=0),
        title={
            "x": 0.5,
            "xanchor": "center",
            "y": 0.8,
            "yanchor": "top",
        },
    )
    coloraxis_name = config["coloraxis_name"]
    _subfig.update_traces(marker=dict(coloraxis=coloraxis_name))
    if config["discrete"]:
        cb = dict(colorbar)
        cb.setdefault("tickmode", "array")
        if "ticks" not in cb:
            cb["ticks"] = "outside"
        colorscale = continuous_scale
        colorbar_args = dict(colorbar=cb, colorscale=colorscale)
        tickvals = cb.get("tickvals") or []
        if tickvals:
            colorbar_args["cmin"] = min(tickvals)
            colorbar_args["cmax"] = max(tickvals)
        _subfig.update_layout(**{coloraxis_name: colorbar_args})
    else:
        _subfig.update_layout(
            **{coloraxis_name: dict(colorbar=colorbar, colorscale=continuous_scale)}
        )


def make_pretty(fig, first_fig, second_fig, third_fig, first, second, third):
    # style the individual subplots first so layout objects carry the right settings
    _make_pretty(first_fig, first)
    _make_pretty(second_fig, second)
    _make_pretty(third_fig, third)

    # position independent colorbars for each subplot
    ca1 = getattr(first_fig["fig"].layout, "coloraxis", None) or getattr(
        first_fig["fig"].layout, "coloraxis1", None
    )
    ca2 = getattr(second_fig["fig"].layout, "coloraxis2", None) or getattr(
        second_fig["fig"].layout, "coloraxis", None
    )
    ca3 = getattr(third_fig["fig"].layout, "coloraxis3", None) or getattr(
        third_fig["fig"].layout, "coloraxis", None
    )
    if ca1 is not None and ca1.colorbar is not None:
        ca1.colorbar.x = 0.35
        ca1.colorbar.len = 0.5
        ca1.colorbar.xanchor = "center"
        ca1.colorbar.thickness = 10
        ca1.colorbar.thicknessmode = "pixels"
    if ca2 is not None and ca2.colorbar is not None:
        ca2.colorbar.x = 0.70
        ca2.colorbar.len = 0.5
        ca2.colorbar.xanchor = "center"
        ca2.colorbar.thickness = 10
        ca2.colorbar.thicknessmode = "pixels"
    if ca3 is not None and ca3.colorbar is not None:
        ca3.colorbar.x = 1.05
        ca3.colorbar.len = 0.5
        ca3.colorbar.xanchor = "center"
        ca3.colorbar.thickness = 10
        ca3.colorbar.thicknessmode = "pixels"

    def _scene_aspect(cfg):
        scene_kwargs = {}
        if cfg.get("aspectmode"):
            scene_kwargs["aspectmode"] = cfg["aspectmode"]
        if cfg.get("aspectratio"):
            scene_kwargs["aspectratio"] = cfg["aspectratio"]
        return scene_kwargs

    scene_overrides = {}
    s1 = _scene_aspect(first)
    if s1:
        scene_overrides["scene"] = s1
    s2 = _scene_aspect(second)
    if s2:
        scene_overrides["scene2"] = s2
    s3 = _scene_aspect(third)
    if s3:
        scene_overrides["scene3"] = s3
    fig.update_layout(
        scene_camera=first_fig["fig"].layout.scene.camera,
        scene2_camera=second_fig["fig"].layout.scene.camera,
        scene3_camera=third_fig["fig"].layout.scene.camera,
        coloraxis=ca1,
        coloraxis2=ca2,
        coloraxis3=ca3,
        legend=first_fig["fig"].layout.legend,
        margin=first_fig["fig"].layout.margin,
        template="none",
        font=dict(family="serif", size=10),
        height=400,
        width=900,
        **scene_overrides,
    )
    fig.update_scenes(
        xaxis_showticklabels=False,
        xaxis_ticks="",
        xaxis_title_text="PC 1",
        yaxis_showticklabels=False,
        yaxis_ticks="",
        yaxis_title_text="PC 2",
        zaxis_showticklabels=False,
        zaxis_ticks="",
        zaxis_title_text="PC 3",
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        scene2=dict(
            xaxis_title="PC 1",
            yaxis_title="",
            zaxis_title="PC 3",
        ),
        scene3=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
    )
    # pull subplot titles closer to the plots
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.update(y=0.75, yanchor="bottom")

    fig.add_annotation(
        x=0.61,
        y=0.2,
        xref="paper",
        yref="paper",
        text="PC 2",
        showarrow=False,
        font=dict(size=12),
    )

    # propagate marker sizes and coloraxis names to the combined figure traces
    trace_start = 0
    for cfg, sub in (
        (first, first_fig),
        (second, second_fig),
        (third, third_fig),
    ):
        trace_end = trace_start + len(sub["fig"].data)
        for trace in fig.data[trace_start:trace_end]:
            _name = trace.name or ""
            if _name.endswith("key"):
                trace.marker.size = cfg["marker_size"]["key"]
            elif _name.endswith("query"):
                trace.marker.size = cfg["marker_size"]["query"]
            trace.showlegend = False
            trace.marker.coloraxis = cfg["coloraxis_name"]
        trace_start = trace_end

    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="Key",
            marker=dict(
                symbol="circle",
                size=10,
                color="black",
            ),
            showlegend=True,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="Query",
            marker=dict(
                symbol="cross",
                size=10,
                color="black",
            ),
            showlegend=True,
        ),
        row=1,
        col=3,
    )

    fig.write_image(os.path.join(BASE_DIR, f"figures/pcas_and_umap.pdf"))
    # save html
    fig.write_html(os.path.join(BASE_DIR, f"figures/pcas_and_umap.html"))
    fig.show()


# %%

first = {
    "C_pos": C_order_pos[(16, 21)],
    "C_neg": C_order_neg[(16, 21)],
    "q_vecs": query_vecs_order[(16, 21)][:2000],
    "k_vecs": key_vecs_order[(16, 21)][:2000],
    "labels": all_order_labels[:2000],
    "q_labels": q_labels[:2000],
    "head": (16, 21),
}


second = {
    "C_pos": C_lex_pos[(16, 21)],
    "C_neg": C_lex_neg[(16, 21)],
    "q_vecs": query_vecs_lex_pos[(16, 21)][:2000],
    "k_vecs": key_vecs_lex_pos[(16, 21)][:2000],
    "labels": all_lexical_labels[:2000],
    "q_labels": all_lexical_labels[:2000],
    "head": (16, 21),
}


third = {
    "C_pos": C_lex_pos[(16, 21)],
    "C_neg": C_lex_neg[(16, 21)],
    "q_vecs": query_vecs_lex_pos[(16, 21)][:1000],
    "k_vecs": key_vecs_lex_pos[(16, 21)][:1000],
    "labels": all_lexical_labels[:1000],
    "q_labels": all_lexical_labels[:1000],
    "head": (16, 21),
    "umap_args": {
        "n_neighbors": 500,
        "min_dist": 0.7,
        "n_components": 3,
        "metric": "cosine",
        "random_state": 99,
        "plot_scale": 0.1,
    },
}


fig, first_fig, second_fig, third_fig = make_plots(
    first,
    second,
    third,
)

# %%

first_make_pretty = {
    "camera": {
        "x": 0.75,
        "y": 0.6,
        "z": 0.5,
        "scale": 3.2,
    },
    "marker_size": {
        "key": 1,
        "query": 3,
    },
    "discrete": False,
    "coloraxis_name": "coloraxis",
}
second_make_pretty = {
    "camera": {
        "x": -0.25,
        "y": 0.4,
        "z": 0.1,
        "scale": 6,
    },
    "keys_only": False,
    "marker_size": {
        "key": 1,
        "query": 2,
    },
    "discrete": True,
    "coloraxis_name": "coloraxis2",
}
third_make_pretty = {
    "camera": {
        "x": 0.4,
        "y": -0.4,
        "z": 0.25,
        "scale": 4.8,
    },
    "aspectmode": "cube",
    "aspectratio": {"x": 1, "y": 1, "z": 1},
    "keys_only": False,
    "marker_size": {
        "key": 1,
        "query": 2,
    },
    "discrete": True,
    "coloraxis_name": "coloraxis3",
}
make_pretty(
    fig,
    first_fig,
    second_fig,
    third_fig,
    first_make_pretty,
    second_make_pretty,
    third_make_pretty,
)
