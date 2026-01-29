from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from box import Box

from toy.dgp import LatentRecallDGP
from toy.recall import Attention
from util_funcs import set_seed


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return {}


def load_model(cfg: Box, run_dir: Path) -> torch.nn.Module:
    model = Attention(cfg.d_model, cfg.d_head, cfg.P).to(cfg.device)
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    weights = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(weights)
    return model


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def compute_G(model, dgp) -> torch.Tensor:
    B = torch.cat([dgp.B1, dgp.B2], dim=1)
    A = torch.cat([dgp.A1, dgp.A2], dim=1)
    WQ = model.W_Q.weight
    WK = model.W_K.weight
    MQ = WQ @ B
    MK = WK @ A
    return MQ.T @ MK


def plot_G_grid(Gs, Ns, titles, nrows=2, ncols=6, cmap="RdBu", center=0.0):
    max_abs = max(float(G.abs().max()) for G in Gs)
    norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=center, vmax=max_abs)
    fig = plt.figure(figsize=(2.5 * ncols, 2.5 * nrows))
    gap_after = 3
    gap_ratio = 0.06
    width_ratios = [1.0] * ncols
    width_ratios.insert(gap_after, gap_ratio)
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.2,
    )
    axes = []
    for r in range(nrows):
        for c in range(ncols):
            col_index = c if c < gap_after else c + 1
            axes.append(fig.add_subplot(gs[r, col_index]))
    im = None
    for ax, G, N, title in zip(axes, Gs, Ns, titles):
        X = G.detach().cpu()
        im = ax.imshow(X, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=15, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(Gs) :]:
        ax.axis("off")

    if nrows >= 2 and ncols >= 4:
        left_col_ax = axes[gap_after - 1]
        right_col_ax = axes[gap_after]
        x_sep = 0.5 * (left_col_ax.get_position().x1 + right_col_ax.get_position().x0)
        top_row_ax = axes[0]
        bottom_row_ax = axes[ncols]
        y_sep = 0.515 * (top_row_ax.get_position().y0 + bottom_row_ax.get_position().y1)
        x0 = min(ax.get_position().x0 for ax in axes)
        x1 = max(ax.get_position().x1 for ax in axes)
        y0 = min(ax.get_position().y0 for ax in axes)
        y1 = max(ax.get_position().y1 for ax in axes)
        fig.add_artist(
            plt.Line2D(
                [x_sep, x_sep],
                [y0, y1],
                transform=fig.transFigure,
                linestyle="--",
                linewidth=0.8,
                color="0.3",
            )
        )
        fig.add_artist(
            plt.Line2D(
                [x0, x1],
                [y_sep, y_sep],
                transform=fig.transFigure,
                linestyle="--",
                linewidth=0.8,
                color="0.3",
            )
        )

    if im is not None:
        fig.tight_layout(pad=0.3, rect=[0.0, 0.0, 0.9, 1.0])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    else:
        fig.tight_layout(pad=0.3)
    return fig


def configure_latex_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def parse_title_from_run_dir(run_dir: Path) -> str:
    match = re.search(r"N(\d+)_M(\d+)_dh(\d+)", run_dir.name)
    if not match:
        return run_dir.name
    n_val, m_val, d_head = match.groups()
    return rf"$d_{{head}}$: {d_head}, $r_1$: {n_val}, $r_2$: {m_val}"


def main():
    configure_latex_plot_style()
    # Point this to the directory that contains your run subfolders
    run_root = Path(".")

    # List the subdirectories you want to plot (in row-major order)
    run_dir_names = [
        "toy_model_checkpoints/dhead_16/N3_M5_dh16_seed11",
    ]

    run_dirs = [run_root / name for name in run_dir_names]
    missing = [p for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run dirs: {missing}")

    Gs = []
    Ns = []
    titles = []
    for run_dir in run_dirs:
        cfg = Box(load_config(run_dir))
        if not cfg:
            print(f"Skipping {run_dir} (missing or invalid config.json)")
            continue

        set_seed(cfg.seed)
        if "device" not in cfg:
            cfg.device = default_device()

        model = load_model(cfg, run_dir)
        dgp = LatentRecallDGP(cfg)
        G = compute_G(model, dgp)
        Gs.append(G)
        Ns.append(cfg.N)
        titles.append(parse_title_from_run_dir(run_dir))

    if not Gs:
        raise RuntimeError("No valid runs found to plot.")

    plot_G_grid(Gs, Ns, titles, nrows=2, ncols=6, cmap="RdBu", center=0.0)
    plt.savefig("G_interactions.pdf")


if __name__ == "__main__":
    main()
