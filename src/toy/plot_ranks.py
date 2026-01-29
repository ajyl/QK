#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import numpy as np
from box import Box

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.collections import PolyCollection
from matplotlib import colors
from toy.dgp import LatentRecallDGP
from toy.recall import Attention, estimate_deltaC
from util_funcs import effective_rank_from_svals, set_seed


RUN_DIR_RE = re.compile(r"N(?P<N>\d+)_M(?P<M>\d+)_dh(?P<dh>\d+)_seed(?P<seed>\d+)")


def load_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return {}


def load_model(cfg, run_dir: Path) -> torch.nn.Module:
    model = Attention(cfg.d_model, cfg.d_head, cfg.P).to(cfg.device)
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    weights = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(weights)
    return model


def infer_run_info(run_dir: Path, cfg: dict[str, Any]) -> dict[str, int]:
    if cfg:
        return {
            "N": int(cfg.get("N", 0)),
            "M": int(cfg.get("M", 0)),
            "d_head": int(cfg.get("d_head", 0)),
            "seed": int(cfg.get("seed", 0)),
        }
    match = RUN_DIR_RE.search(run_dir.name)
    if not match:
        return {"N": 0, "M": 0, "d_head": 0, "seed": 0}
    return {
        "N": int(match.group("N")),
        "M": int(match.group("M")),
        "d_head": int(match.group("dh")),
        "seed": int(match.group("seed")),
    }


def load_delta(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def load_accuracy(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return "NA"
    try:
        with metrics_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if not row:
                return "NA"
            acc = row.get("accuracy", "")
            if not acc:
                return "NA"
            return f"{float(acc):.4f}"
    except (ValueError, OSError, csv.Error):
        return "NA"


def format_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        "N",
        "M",
        "d_head",
        "exp_rank_z1",
        "obs_rank_z1",
        "exp_rank_z2",
        "obs_rank_z2",
        "accuracy",
    ]
    widths: dict[str, int] = {}
    for col in columns:
        widths[col] = max(len(col), max(len(str(r[col])) for r in rows))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    lines = [header, sep]
    for row in rows:
        line = "  ".join(str(row[col]).ljust(widths[col]) for col in columns)
        lines.append(line)
    return lines


def build_rank_tri_grid(rows: list[dict[str, Any]], idx=None, out_dir=None) -> tuple[
    list[int],
    list[int],
    dict[tuple[int, int], list[int]],
    dict[tuple[int, int], list[int]],
]:
    exp_z1_vals = sorted({r["exp_rank_z1"] for r in rows})
    exp_z2_vals = sorted({r["exp_rank_z2"] for r in rows})
    if idx is not None:
        if "dhead_8" in out_dir.name:
            exp_z1_vals = exp_z1_vals[:-1]
            exp_z2_vals = exp_z2_vals[:-1]
        elif "dhead_16" in out_dir.name:
            exp_z1_vals = exp_z1_vals[1:]
            exp_z2_vals = exp_z2_vals[1:]
    obs_z1: dict[tuple[int, int], list[int]] = {}
    obs_z2: dict[tuple[int, int], list[int]] = {}

    for r in rows:
        key = (r["exp_rank_z1"], r["exp_rank_z2"])
        obs_z1.setdefault(key, []).append(int(r["obs_rank_z1"]))
        obs_z2.setdefault(key, []).append(int(r["obs_rank_z2"]))

    return exp_z1_vals, exp_z2_vals, obs_z1, obs_z2


def plot_rank_tri_heatmap(
    exp_z1_vals: list[int],
    exp_z2_vals: list[int],
    obs_z1: dict[tuple[int, int], list[int]],
    obs_z2: dict[tuple[int, int], list[int]],
    title: str,
    ax: plt.Axes,
    idx: int,
    norm: colors.Normalize | None = None,
    ylabel_pad: float | None = None,
) -> None:

    def mean_or_none(vals: list[int]) -> float | None:
        if not vals:
            return None
        return float(sum(vals)) / len(vals)

    def text_color(val: float) -> str:
        return "white" if norm(val) < 0.25 else "black"
        #return "black"

    def format_label(val: float) -> str:
        return str(int(round(val)))

    if norm is None:
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap_base = plt.get_cmap("YlGn_r")
    cmap = colors.LinearSegmentedColormap.from_list(
        "YlGn_r_mid",
        cmap_base(np.linspace(0.35, 0.85, 256)),
    )

    polys = []
    facecolors = []
    text_specs: list[tuple[float, float, str, str]] = []
    exp_z1_plot = list(reversed(exp_z1_vals))
    for i, exp_z1 in enumerate(exp_z1_plot):
        for j, exp_z2 in enumerate(exp_z2_vals):
            x0, x1 = j, j + 1
            y0, y1 = i, i + 1
            key = (exp_z1, exp_z2)

            mean_z1 = mean_or_none(obs_z1.get(key, []))
            if mean_z1 is not None:
                left_top_tri = [(x0, y1), (x1, y1), (x0, y0)]
                diff_z1 = abs(mean_z1 - exp_z1)
                polys.append(left_top_tri)
                facecolors.append(cmap(norm(diff_z1)))
                text_specs.append(
                    (x0 + 0.3, y0 + 0.7, format_label(mean_z1), text_color(diff_z1))
                )

            mean_z2 = mean_or_none(obs_z2.get(key, []))
            if mean_z2 is not None:
                right_bottom_tri = [(x1, y0), (x1, y1), (x0, y0)]
                diff_z2 = abs(mean_z2 - exp_z2)
                polys.append(right_bottom_tri)
                facecolors.append(cmap(norm(diff_z2)))
                text_specs.append(
                    (x0 + 0.7, y0 + 0.3, format_label(mean_z2), text_color(diff_z2))
                )

    collection = PolyCollection(
        polys, facecolors=facecolors, edgecolors="0.85", linewidths=0.4
    )
    ax.add_collection(collection)
    ax.set_xlim(0, len(exp_z2_vals))
    ax.set_ylim(0, len(exp_z1_vals))
    ax.set_xticks([i + 0.5 for i in range(len(exp_z2_vals))])
    ax.set_yticks([i + 0.5 for i in range(len(exp_z1_vals))])
    ax.set_xticklabels(exp_z2_vals)
    ax.set_yticklabels(exp_z1_plot)
    ax.set_xlabel(r"Groundtruth Rank ($r_2$)")

    if ylabel_pad is None:
        ax.set_ylabel(r"Groundtruth Rank ($r_1$)")
    else:
        ax.set_ylabel(r"Groundtruth Rank ($r_1$)", labelpad=ylabel_pad)
    ax.set_title(title)
    ax.set_aspect("equal")

    for x, y, label, color in text_specs:
        ax.text(x, y, label, color=color, ha="center", va="center", fontsize=9)


def apply_latex_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def add_triangle_rank_legend(fig: plt.Figure, cax: plt.Axes) -> None:
    cax_pos = cax.get_position()
    legend_height = cax_pos.height * 0.3
    legend_gap = 0.02
    legend_width = min(0.22, cax_pos.width * 4.0)
    legend_left = max(0.02, cax_pos.x0 + (cax_pos.width - legend_width) / 2)
    legend_width = min(legend_width, 0.98 - legend_left)
    legend_bottom = cax_pos.y1 + legend_gap
    if legend_bottom >= 0.98:
        return
    legend_top = min(0.98, legend_bottom + legend_height)
    legend_height = legend_top - legend_bottom
    if legend_height <= 0:
        return
    legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_aspect("equal", adjustable="box")

    left_top_tri = [(0.05, 0.95), (0.95, 0.95), (0.05, 0.05)]
    right_bottom_tri = [(0.95, 0.05), (0.95, 0.95), (0.05, 0.05)]
    legend_tris = PolyCollection(
        [left_top_tri, right_bottom_tri],
        facecolors=["0.85", "0.7"],
        edgecolors="0.5",
        linewidths=0.4,
    )
    legend_ax.add_collection(legend_tris)
    legend_ax.text(0.32, 0.68, r"$r_1$", ha="center", va="center", fontsize=10)
    legend_ax.text(0.68, 0.32, r"$r_2$", ha="center", va="center", fontsize=10)
    legend_ax.text(
        0.5,
        1.05,
        "Observed\nranks",
        ha="center",
        va="bottom",
        fontsize=9,
        clip_on=False,
    )


def shorten_colorbar_axes(cax: plt.Axes, height_scale: float = 0.78) -> None:
    cax_pos = cax.get_position()
    new_height = cax_pos.height * height_scale
    new_pos = [cax_pos.x0, cax_pos.y0, cax_pos.width, new_height]
    cax.set_position(new_pos)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare expected vs observed effective ranks for delta_z1 and delta_z2."
    )
    parser.add_argument(
        "--out-dirs",
        type=str,
        nargs="+",
        help="One or more root directories containing run subdirectories",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=0.99,
        help="Energy threshold for effective rank",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save heatmap images (default: out-dir)",
        required=True,
    )
    args = parser.parse_args()
    apply_latex_style()

    out_dirs = [Path(p) for p in args.out_dirs]
    save_dir = Path(args.save_dir) if args.save_dir else out_dirs[0]
    save_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[list[dict[str, Any]]] = []
    for out_dir in out_dirs:
        delta_paths = list(out_dir.rglob("delta_z1.pt"))
        if not delta_paths:
            print(f"No delta_z1.pt files found under {out_dir}")
            all_rows.append([])
            continue

        rows: list[dict[str, Any]] = []
        missing = 0
        for delta_z1_path in delta_paths:
            run_dir = delta_z1_path.parent
            if "dhead_16" in out_dir.name and "N3" in run_dir.name:
                continue
            delta_z2_path = run_dir / "delta_z2.pt"
            if not delta_z2_path.exists():
                missing += 1
                continue

            cfg = Box(load_config(run_dir))
            set_seed(cfg.seed)
            model = load_model(cfg, run_dir)
            info = infer_run_info(run_dir, cfg)

            dgp = LatentRecallDGP(cfg)
            delta_z1 = estimate_deltaC(cfg, dgp, model, factor="z1", mean_center=False)
            delta_z2 = estimate_deltaC(cfg, dgp, model, factor="z2", mean_center=False)

            s1 = torch.linalg.svdvals(delta_z1)
            s2 = torch.linalg.svdvals(delta_z2)
            r1 = effective_rank_from_svals(s1, energy=args.energy)
            r2 = effective_rank_from_svals(s2, energy=args.energy)
            acc = load_accuracy(run_dir)

            row = {
                # "run_dir": str(run_dir),
                "N": info["N"],
                "M": info["M"],
                "d_head": info["d_head"],
                # "seed": info["seed"],
                "exp_rank_z1": info["N"],
                "obs_rank_z1": r1,
                "exp_rank_z2": info["M"],
                "obs_rank_z2": r2,
                "accuracy": acc,
            }
            rows.append(row)

        rows.sort(key=lambda r: (r["N"], r["M"], r["d_head"]))
        all_rows.append(rows)

        for line in format_table(rows):
            print(line)

    def max_abs_diff(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 1.0
        exp_z1_vals, exp_z2_vals, obs_z1, obs_z2 = build_rank_tri_grid(rows, out_dir=out_dir)
        all_diffs: list[float] = []
        for (exp_z1, exp_z2), vals in obs_z1.items():
            if vals:
                mean_val = float(sum(vals)) / len(vals)
                all_diffs.append(abs(mean_val - exp_z1))
        for (exp_z1, exp_z2), vals in obs_z2.items():
            if vals:
                mean_val = float(sum(vals)) / len(vals)
                all_diffs.append(abs(mean_val - exp_z2))
        return max(all_diffs) if all_diffs else 1.0

    global_max = max(max_abs_diff(rows) for rows in all_rows) if all_rows else 1.0
    shared_norm = colors.Normalize(vmin=0.0, vmax=global_max)

    n_panels = max(1, len(out_dirs))
    #fig = plt.figure(figsize=(3.2 * n_panels + 0.4, 3))
    fig = plt.figure(figsize=(2.4 * n_panels + 0.2, 1.8))
    width_ratios = [1] * n_panels + [0.04]
    gs = fig.add_gridspec(1, n_panels + 1, width_ratios=width_ratios, wspace=0.4)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(n_panels)]
    cax = fig.add_subplot(gs[0, n_panels])
    shorten_colorbar_axes(cax, height_scale = 0.7)
    for idx, (ax, out_dir, rows) in enumerate(zip(axes, out_dirs, all_rows, strict=False)):
        if not rows:
            ax.set_axis_off()
            ax.set_title(f"No data: {out_dir}")
            continue
        match = re.search(r"(\d+)", out_dir.name)
        d_head = match.group(1) if match else out_dir.name
        exp_z1_vals, exp_z2_vals, obs_z1, obs_z2 = build_rank_tri_grid(rows, idx, out_dir)
        task_name = "(Task 1)"
        if "gaussian" in out_dir.name:
            task_name = "(Task 2)"
        task_var = idx // 2 + 1
        plot_rank_tri_heatmap(
            exp_z1_vals,
            exp_z2_vals,
            obs_z1,
            obs_z2,
            #f"Expected vs. Observed Ranks\n$d_{{head}} = {d_head}$, Task Variant {task_var}",
            #f"$d_{{head}} = {d_head}$, Task Variant {task_var}",
            f"$d_{{head}} = {d_head}$ {task_name}",
            ax,
            idx,
            norm=shared_norm,
            ylabel_pad=1.0 if idx > 0 else None,
        )
    cmap_base = plt.get_cmap("YlGn_r")
    cmap = colors.LinearSegmentedColormap.from_list(
        "YlGn_r_mid",
        cmap_base(np.linspace(0.35, 0.85, 256)),
    )
    mappable = plt.cm.ScalarMappable(norm=shared_norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax, label="Missing Ranks\n(Groundtruth - Observed)", pad=0.01)
    cbar.locator = MaxNLocator(integer=True)
    cbar.formatter = FormatStrFormatter("%d")
    cbar.update_ticks()
    fig.tight_layout(pad=0.4)
    add_triangle_rank_legend(fig, cax)
    filename = "_".join(
        Path(p).name for p in args.out_dirs
    )
    fig.savefig(save_dir / f"rank_heatmap_{filename}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
