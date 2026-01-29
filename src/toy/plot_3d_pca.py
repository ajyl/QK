from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, Tuple

import torch

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from toy.dgp import GaussianTwoFactorRecallDGP, LatentRecallDGP, Config as GaussianDGPConfig
from toy.recall import Attention, estimate_deltaC, Config as RecallConfig
from util_funcs import effective_rank_from_svals, set_seed


def load_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return {}


def load_model(cfg: RecallConfig, run_dir: Path) -> Attention:
    model = Attention(cfg.d_model, cfg.d_head, cfg.P).to(cfg.device)
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    weights = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(weights)
    return model


def apply_cfg_overrides(cfg: RecallConfig, overrides: dict[str, Any]) -> RecallConfig:
    cfg_fields = {f.name for f in fields(RecallConfig)}
    for key, value in overrides.items():
        if key in cfg_fields:
            setattr(cfg, key, value)
    return cfg


def bin_by_quantiles(x: torch.Tensor, bins: int = 8) -> torch.Tensor:
    """
    Return integer bin ids 0..bins-1 by quantiles.
    """
    x = x.detach().cpu()
    qs = torch.quantile(x, torch.linspace(0, 1, bins + 1))
    # make right edge inclusive
    qs[-1] = qs[-1] + 1e-9
    ids = torch.bucketize(x, qs[1:], right=False)
    return ids



def build_dgp(
    cfg: RecallConfig, gaussian_dgp: bool
) -> LatentRecallDGP | GaussianTwoFactorRecallDGP:
    if gaussian_dgp:
        gcfg = GaussianDGPConfig(
            r1=cfg.N,
            r2=cfg.M,
            C=cfg.C,
            P=cfg.P,
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            noise_std=cfg.noise_std,
            device=cfg.device,
        )
        return GaussianTwoFactorRecallDGP(gcfg)
    return LatentRecallDGP(cfg)


@torch.no_grad()
def subspaces_from_deltaC(
    cfg: RecallConfig, DeltaC: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    DeltaC = U Σ V^T. Return U_r, V_r, singular values, r (effective rank).
    """
    U, S, Vh = torch.linalg.svd(DeltaC, full_matrices=False)
    r = effective_rank_from_svals(S, energy=cfg.energy_threshold)
    return U[:, :r], Vh.T[:, :r], S, r


@torch.no_grad()
def build_points_for_pca(
    cfg: RecallConfig,
    dgp: LatentRecallDGP | GaussianTwoFactorRecallDGP,
    model: Attention,
    U_r: torch.Tensor,
    V_r: torch.Tensor,
    num_points: int,
    color_by: Literal["s1_0", "s1_norm"] = "s1_0",
    factor: Literal["z1", "z2"] = "z1",
) -> pd.DataFrame:
    """
    Sample sequences, take q at query position and k at *true target* position,
    project into factor subspaces: cq = U_r^T q, ck = V_r^T k, stack, ->3D.
    Uses PCA for z1 and UMAP for z2.
    """
    device = cfg.device
    model.eval()

    # Collect projected coords and labels
    cq_list, ck_list = [], []
    latent_list = []

    B = 256
    remaining = num_points
    while remaining > 0:
        b = min(B, remaining)
        x_seq, _, info = dgp.sample_batch(b)

        C = cfg.C
        x_ctx = x_seq[:, :C, :]  # [b,C,d]
        x_q = x_seq[:, C, :]  # [b,d]
        i_star = info["i_star"]  # [b]
        if factor == "z1":
            if "s1_star" in info:
                latent_star = info["s1_star"]
            elif "u1_star" in info:
                latent_star = info["u1_star"]
            elif "z1_star" in info:
                latent_star = info["z1_star"]
            else:
                raise KeyError("Expected s1_star/u1_star/z1_star in info dict.")
        else:
            if "s2_star" in info:
                latent_star = info["s2_star"]
            elif "u2_star" in info:
                latent_star = info["u2_star"]
            elif "z2_star" in info:
                latent_star = info["z2_star"]
            else:
                raise KeyError("Expected s2_star/u2_star/z2_star in info dict.")

        # gather the target context vector and map to k
        x_k_tgt = x_ctx[torch.arange(b, device=device), i_star]  # [b,d]
        q, k = model.get_qk(x_q, x_k_tgt)  # [b,dh]

        # project into extracted subspaces (r dims)
        cq = q @ U_r  # [b,r] because U_r is [dh,r]
        ck = k @ V_r  # [b,r]

        cq_list.append(cq)
        ck_list.append(ck)
        latent_list.append(latent_star)

        remaining -= b

    cq = torch.cat(cq_list, dim=0)  # [n,r]
    ck = torch.cat(ck_list, dim=0)  # [n,r]
    latent = torch.cat(latent_list, dim=0)  # [n,r]

    # Stack (queries + keys) and reduce to 3D
    X = torch.cat([cq, ck], dim=0)  # [2n,r]
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X.detach().cpu().numpy())
    data_type = ["Query"] * cq.shape[0] + ["Key"] * ck.shape[0]

    # Color labels
    if color_by is None:
        latent_cpu = latent.detach().cpu()
        latent_codes = [
            ",".join("1" if v > 0 else "-1" for v in row.tolist()) for row in latent_cpu
        ]
        color = latent_codes + latent_codes
        color_name = f"{factor}_code"
    elif color_by == "s1_0":
        c_raw = latent[:, 1]
        c_bin = bin_by_quantiles(c_raw, bins=8)
        color = torch.cat([c_bin, c_bin], dim=0).numpy()
        color_name = f"{factor}_0_bin"
    else:
        c_raw = torch.norm(latent, dim=-1)
        c_bin = bin_by_quantiles(c_raw, bins=8)
        color = torch.cat([c_bin, c_bin], dim=0).numpy()
        color_name = f"{factor}_norm_bin"


    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "PC3": pca_result[:, 2],
            "DataType": data_type,
            color_name: color,
        }
    )
    return df


def make_plot(
    df: pd.DataFrame,
    color_col: str,
    factor: Literal["z1", "z2"] = "z1",
    connect_key_means_hypercube: bool = False,
) -> go.Figure:

    key_marker_size = 1.5
    query_marker_size = 4

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_col,
        symbol="DataType",
        symbol_map={"Key": "circle", "Query": "cross"},
        labels={
            "DataType": "Query vs. Key",
            color_col: "S_1[0]<br>(Quantized)"
        },
    )
    fig.update_layout(legend_title_text="Query vs. Key")

    code_col = f"{factor}_code"
    if connect_key_means_hypercube and code_col in df.columns:
        key_df = df[df["DataType"] == "Key"]
        means = key_df.groupby(code_col)[["PC1", "PC2", "PC3"]].mean().reset_index()
        code_to_mean = {
            row[code_col]: (row["PC1"], row["PC2"], row["PC3"])
            for _, row in means.iterrows()
        }
        code_to_bits = {
            code: tuple(int(v) for v in code.split(",")) for code in code_to_mean
        }
        bits_to_code = {bits: code for code, bits in code_to_bits.items()}

        x_line, y_line, z_line = [], [], []
        added = set()
        for code, bits in code_to_bits.items():
            for i in range(len(bits)):
                neighbor_bits = list(bits)
                neighbor_bits[i] *= -1
                neighbor_bits = tuple(neighbor_bits)
                neighbor_code = bits_to_code.get(neighbor_bits)
                if neighbor_code is None:
                    continue
                edge_key = tuple(sorted([code, neighbor_code]))
                if edge_key in added:
                    continue
                added.add(edge_key)
                x0, y0, z0 = code_to_mean[code]
                x1, y1, z1 = code_to_mean[neighbor_code]
                x_line.extend([x0, x1, None])
                y_line.extend([y0, y1, None])
                z_line.extend([z0, z1, None])

        if x_line:
            fig.add_trace(
                go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="key mean hypercube",
                    showlegend=True,
                )
            )

    fig.update_layout(legend_title_text="Z1 Code")
    for i, trace in enumerate(fig.data):
        trace_name = trace.name or ""
        symbol = trace.marker.symbol
        if "Key" in trace_name or symbol == "circle":
            fig.data[i].marker.size = key_marker_size
        elif "Query" in trace_name or symbol == "cross":
            fig.data[i].marker.size = query_marker_size
        trace.showlegend = False
        trace.legend = "legend"

    show_legend = True
    if factor == "z2":
        show_legend = False
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="Key",
            marker=dict(
                symbol="circle",
                size=6,
                color="black",
            ),
            showlegend=show_legend,
            legend="legend",
        ),
        row=1,
        col=1,
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
                size=6,
                color="black",
            ),
            showlegend=show_legend,
            legend="legend",
        ),
        row=1,
        col=1,
    )

    category_colors = {}
    for trace in fig.data:
        category = trace.name.split(" ")[0]
        if category.endswith(","):
            category = category[:-1]
        if not category.endswith("1"):
            continue
        if category not in category_colors:
            category_colors[category] = trace.marker.color

    for category, color in category_colors.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=category,
                marker=dict(
                    symbol="circle",
                    size=6,
                    color=color,
                ),
                showlegend=show_legend,
                legend="legend",
            ),
            row=1,
            col=1,
        )

    if show_legend:
        legend_items = [
            ("Key", "black", "circle"),
            ("Query", "black", "cross"),
        ] + [(category, color, "circle") for category, color in category_colors.items()]

        legend_x = 0.93
        legend_y = 0.475
        line_height = 0.06
        pad = 0.015
        box_width = 0.27
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
            line=dict(color="rgba(0,0,0,0.8)", width=1),
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
                    x=box_x0 + 0.075,
                    y=y_pos,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    text=label,
                    font=dict(size=9, color="black"),
                )
            )
        fig.update_layout(annotations=annotations, showlegend=False)
    else:
        fig.update_layout(showlegend=False)

    camera_scale = 0.95
    fig.update_layout(
        title=dict(
            #text="z<sub>1</sub> Subspace (Rank 3)",
            text="s<sub>1</sub> Subspace (Rank 3)",
            x=0.55,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(family="Times New Roman, Times, serif", size=10),
        ),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            #camera=dict(
            #    eye=dict(
            #        x=0.85 * camera_scale, y=-1.0 * camera_scale, z=1.4 * camera_scale
            #    ),
            #),
        ),
        font=dict(family="Times New Roman, Times, serif", size=9),
        margin=dict(l=30, r=5, t=0, b=30),
        width=280,
        height=240,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # Update legend title:

    return fig


# -----------------------------
# Main
# -----------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing config.json and model.pt.",
    )
    p.add_argument(
        "--num_points",
        type=int,
        default=1600,
        help="Number of query-key pairs to visualize.",
    )
    p.add_argument(
        "--connect_key_means_hypercube",
        action="store_true",
        help="Connect mean Key vectors across discrete latent hypercube edges.",
    )
    p.add_argument(
        "--factor",
        type=str,
        default="z1",
        choices=["z1", "z2"],
        help="Which latent factor to visualize.",
    )
    p.add_argument(
        "--gaussian-dgp",
        action="store_true",
        help="Use GaussianTwoFactorRecallDGP for sampling.",
    )
    p.add_argument(
        "--latent-dgp",
        action="store_true",
        help="Force LatentRecallDGP even if run-dir suggests gaussian.",
    )
    p.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Optional path to save plotly HTML.",
    )
    p.add_argument(
        "--color_by",
        type=str,
        default=None,
        choices=[None, "s1_0", "s1_norm"],
        help="Color by first coordinate or norm of the selected factor.",
    )
    args = p.parse_args()

    cfg = RecallConfig()
    dgp: LatentRecallDGP | GaussianTwoFactorRecallDGP
    model: Attention

    gaussian_dgp = args.gaussian_dgp
    if args.latent_dgp:
        gaussian_dgp = False

    run_dir = Path(args.run_dir)
    cfg = apply_cfg_overrides(cfg, load_config(run_dir))
    set_seed(cfg.seed)
    if not gaussian_dgp and "gaussian" in str(run_dir).lower():
        gaussian_dgp = True
    dgp = build_dgp(cfg, gaussian_dgp)
    model = load_model(cfg, run_dir)
    print(f"Loaded run: {run_dir}")

    Delta = estimate_deltaC(cfg, dgp, model, factor=args.factor, mean_center=False)
    U_r, V_r, S, r = subspaces_from_deltaC(cfg, Delta)
    print(
        f"ΔC_{args.factor}: top singular values = {S[:min(10, S.numel())].detach().cpu().numpy()}"
    )
    print(f"ΔC_{args.factor} effective rank @ {cfg.energy_threshold:.2f} = {r}")

    df = build_points_for_pca(
        cfg,
        dgp,
        model,
        U_r,
        V_r,
        num_points=args.num_points,
        color_by=args.color_by,
        factor=args.factor,
    )

    # choose color column
    if args.color_by is None:
        color_col = f"{args.factor}_code"
    elif args.color_by == "s1_0":
        color_col = f"{args.factor}_0_bin"
    else:
        color_col = f"{args.factor}_norm_bin"

    fig = make_plot(
        df,
        color_col=color_col,
        factor=args.factor,
        connect_key_means_hypercube=args.connect_key_means_hypercube,
    )

    fig.write_html(f"{args.save_path}.html")
    fig.write_image(f"{args.save_path}.pdf")
    print(f"Saved plot to {args.save_path}.html and .pdf")


if __name__ == "__main__":
    main()
