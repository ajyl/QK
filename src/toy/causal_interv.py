from __future__ import annotations

import math
from typing import Literal, Tuple, Dict, Optional
from pathlib import Path
import json
from dataclasses import fields
import itertools

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
def _orthonormalize_cols(X: torch.Tensor) -> torch.Tensor:
    # QR gives orthonormal columns (reduced)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q


@torch.no_grad()
def get_qk_headspace(
    model, x_ctx: torch.Tensor, x_q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_ctx: [B, C, d_model]
    x_q:   [B, d_model]
    returns:
      q: [B, d_head]
      k: [B, C, d_head]
    """
    if hasattr(model, "ln") and model.ln is not None:
        x_ctx = model.ln(x_ctx)
        x_q = model.ln(x_q)

    q = model.W_Q(x_q)  # [B, dh]
    k = model.W_K(x_ctx)  # [B, C, dh]
    return q, k


@torch.no_grad()
def attention_probs(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    q: [B, dh], k: [B, C, dh] -> attn: [B, C]
    """
    dh = q.shape[-1]
    scores = torch.einsum("bd,bcd->bc", q, k) / math.sqrt(dh)
    return torch.softmax(scores, dim=-1)


@torch.no_grad()
def random_orthonormal_subspace(
    dh: int,
    r: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Return V in R^{dh x r} with orthonormal columns (Haar-ish via QR of Gaussian).
    """
    X = torch.randn(dh, r, device=device, generator=generator)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q


@torch.no_grad()
def make_random_bases(
    dh: int,
    r: int,
    n_trials: int,
    device: torch.device,
    *,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """
    Pre-generate random orthonormal bases to reuse across batches (lower variance).
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    return [
        random_orthonormal_subspace(dh, r, device, generator=gen)
        for _ in range(n_trials)
    ]


def _avg_dicts(dicts: list[dict]) -> dict:
    """
    Average numeric values across a list of metric dicts (assumes same keys).
    """
    out = {}
    keys = dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in dicts]
        if isinstance(vals[0], (int, float)):
            out[k] = float(sum(vals) / len(vals))
        else:
            out[k] = vals[0]
    return out


@torch.no_grad()
def select_itarget_by_u_mismatch(
    info: dict,
    i_orig: torch.Tensor,  # [B]
    factor: Literal["z1", "z2", "both"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      i_tgt: [B_valid]
      keep:  [B] boolean mask for which examples had at least one valid target.

    Requires:
      info["u1"]:      [B, C, N]
      info["u2"]:      [B, C, M]
      info["u1_star"]: [B, N]
      info["u2_star"]: [B, M]

    Rules:
      z1:   match u2_star AND NOT match u1_star
      z2:   match u1_star AND NOT match u2_star
      both: NOT match u1_star AND NOT match u2_star

    Excludes i_orig from candidates.
    """
    u1 = info["u1"]  # [B,C,N]
    u2 = info["u2"]  # [B,C,M]
    u1s = info["u1_star"]  # [B,N]
    u2s = info["u2_star"]  # [B,M]

    B, C, _ = u1.shape
    device = u1.device
    b_idx = torch.arange(B, device=device)

    match_u1 = (u1 == u1s[:, None, :]).all(dim=-1)  # [B,C]
    match_u2 = (u2 == u2s[:, None, :]).all(dim=-1)  # [B,C]

    if factor == "z1":
        cand = match_u2 & (~match_u1)
    elif factor == "z2":
        cand = match_u1 & (~match_u2)
    elif factor == "both":
        cand = (~match_u1) & (~match_u2)
    else:
        raise ValueError("factor must be 'z1', 'z2', or 'both'")

    # exclude i_orig
    cand[b_idx, i_orig] = False

    keep = cand.any(dim=-1)  # [B]
    if keep.sum().item() == 0:
        # nothing usable in this proposal batch
        return torch.empty(0, dtype=torch.long, device=device), keep

    # sample i_target uniformly among candidates per kept row
    cand_kept = cand[keep].float()  # [B_keep, C]
    probs = cand_kept / cand_kept.sum(dim=-1, keepdim=True)
    i_tgt = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B_keep]

    return i_tgt, keep


@torch.no_grad()
def sample_valid_batch_with_targets(
    cfg,
    dgp,
    model,
    *,
    want_B: int,
    factor: Literal["z1", "z2", "both"],
    choose_i_orig: Literal["i_star", "argmax"] = "i_star",
    proposal_B: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_ctx: [want_B, C, d_model]
      x_q:   [want_B, d_model]
      info:  filtered info dict (only keys present are filtered)
      i_orig:[want_B]
      i_tgt: [want_B]
    """
    device = torch.device(cfg.device)
    C = cfg.C

    kept_x_ctx = []
    kept_x_q = []
    kept_i_orig = []
    kept_i_tgt = []
    kept_info = {}  # build lists then cat

    n_kept = 0
    while n_kept < want_B:
        x_seq, _, info = dgp.sample_batch(proposal_B)  # x_seq [B, C+1, d]
        x_ctx = x_seq[:, :C, :]
        x_q = x_seq[:, C, :]

        # choose i_orig
        if choose_i_orig == "i_star":
            if "i_star" not in info:
                raise KeyError("choose_i_orig='i_star' but info['i_star'] not found.")
            i_orig = info["i_star"]  # [B]
        else:
            # compute argmax attention to pick what model currently attends to
            # assumes your model has W_Q/W_K and (optional) ln
            q, k = get_qk_headspace(model, x_ctx, x_q)  # q [B,dh], k [B,C,dh]
            attn = attention_probs(q, k)  # [B,C]
            i_orig = attn.argmax(dim=-1)

        # select i_target with NO fallback and compute keep mask
        i_tgt_kept, keep = select_itarget_by_u_mismatch(info, i_orig, factor=factor)
        if keep.sum().item() == 0:
            continue

        # filter tensors
        x_ctx_kept = x_ctx[keep]
        x_q_kept = x_q[keep]
        i_orig_kept = i_orig[keep]

        # align i_tgt_kept already corresponds to kept rows
        assert x_ctx_kept.shape[0] == i_tgt_kept.shape[0]

        # take as many as needed
        need = want_B - n_kept
        take = min(need, x_ctx_kept.shape[0])

        kept_x_ctx.append(x_ctx_kept[:take])
        kept_x_q.append(x_q_kept[:take])
        kept_i_orig.append(i_orig_kept[:take])
        kept_i_tgt.append(i_tgt_kept[:take])

        # filter info dict entries that are tensors with first dim = proposal_B
        for kname, v in info.items():
            if torch.is_tensor(v) and v.shape[0] == proposal_B:
                kept_info.setdefault(kname, []).append(v[keep][:take])

        n_kept += take

    # concatenate
    x_ctx_out = torch.cat(kept_x_ctx, dim=0).to(device)
    x_q_out = torch.cat(kept_x_q, dim=0).to(device)
    i_orig_out = torch.cat(kept_i_orig, dim=0).to(device)
    i_tgt_out = torch.cat(kept_i_tgt, dim=0).to(device)

    info_out = {k: torch.cat(vs, dim=0).to(device) for k, vs in kept_info.items()}
    return x_ctx_out, x_q_out, info_out, i_orig_out, i_tgt_out


@torch.no_grad()
def swap_key_coords_in_subspace(
    k: torch.Tensor,  # [B, C, dh]
    i_orig: torch.Tensor,  # [B]
    i_tgt: torch.Tensor,  # [B]
    V: torch.Tensor,  # [dh, r] with orthonormal cols
) -> torch.Tensor:
    """
    Implements your LaTeX swap in span(V) (assuming V has orthonormal columns):
      k_orig <- k_orig + V (V^T k_tgt - V^T k_orig)
      k_tgt  <- k_tgt  + V (V^T k_orig - V^T k_tgt)
    """
    B, C, dh = k.shape
    b = torch.arange(B, device=k.device)

    k_new = k.clone()

    k_orig = k_new[b, i_orig]  # [B, dh]
    k_tgt = k_new[b, i_tgt]  # [B, dh]

    # coords in subspace: [B, r]
    c_orig = k_orig @ V
    c_tgt = k_tgt @ V

    # move vectors along the subspace directions
    delta_orig = (c_tgt - c_orig) @ V.T  # [B, dh]
    delta_tgt = (c_orig - c_tgt) @ V.T  # [B, dh]

    k_new[b, i_orig] = k_orig + delta_orig
    k_new[b, i_tgt] = k_tgt + delta_tgt

    return k_new


@torch.no_grad()
def run_key_swap_experiment(
    cfg,
    dgp,
    model,
    V1: torch.Tensor,  # [dh, r1]
    V2: torch.Tensor,  # [dh, r2]
    *,
    n_batches: int = 200,
    batch_size: int = 256,
    random_baseline_trials: int = 10,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    For each batch:
      - compute baseline attention
      - choose i_orig (either current argmax attention or i_star)
      - sample i_target != i_orig
      - apply key swap in z1, z2, or both
      - measure attention shift from i_orig -> i_target

    Returns aggregated DataFrame over batches with:
      condition, attn_on_orig, attn_on_tgt, shift(tgt-orig), tgt_argmax_rate
    """
    device = cfg.device
    C = cfg.C
    dh = cfg.d_head

    model.eval()

    Vboth = torch.cat([V1, V2], dim=1)
    Vboth = _orthonormalize_cols(Vboth)

    r1 = V1.shape[1]
    r2 = V2.shape[1]
    r_both = Vboth.shape[1]

    rand_V1_list = make_random_bases(
        dh, r1, random_baseline_trials, device, seed=random_seed + 101
    )
    rand_V2_list = make_random_bases(
        dh, r2, random_baseline_trials, device, seed=random_seed + 202
    )
    rand_Vb_list = make_random_bases(
        dh, r_both, random_baseline_trials, device, seed=random_seed + 303
    )

    rows = []
    for _ in range(n_batches):
        x_ctx, x_q, info_filt, i_orig, i_tgt = sample_valid_batch_with_targets(
            cfg,
            dgp,
            model,
            want_B=batch_size,
            factor="both",
            choose_i_orig="i_star",
            proposal_B=4096,
        )

        q, k = get_qk_headspace(model, x_ctx, x_q)  # q: [B,dh], k: [B,C,dh]
        attn0 = attention_probs(q, k)  # [B,C]

        def metrics_from_attn(
            attn: torch.Tensor, i_orig: torch.Tensor, i_tgt: torch.Tensor
        ) -> dict:
            b = torch.arange(attn.shape[0], device=attn.device)
            a_orig = attn[b, i_orig]
            a_tgt = attn[b, i_tgt]
            top = attn.argmax(dim=-1)
            return {
                "attn_orig_mean": a_orig.mean().item(),
                "attn_tgt_mean": a_tgt.mean().item(),
                "shift_mean": (a_tgt - a_orig).mean().item(),
                "tgt_argmax_rate": (top == i_tgt).float().mean().item(),
                "orig_argmax_rate": (top == i_orig).float().mean().item(),
            }

        # baseline (no intervention)
        m0 = metrics_from_attn(attn0, i_orig, i_tgt)
        rows.append({"condition": "Null", **m0})

        # ---- actual subspace swaps ----
        k_z1 = swap_key_coords_in_subspace(k, i_orig, i_tgt, V1)
        rows.append(
            {
                "condition": "$z_1$",
                **metrics_from_attn(attention_probs(q, k_z1), i_orig, i_tgt),
            }
        )

        k_z2 = swap_key_coords_in_subspace(k, i_orig, i_tgt, V2)
        rows.append(
            {
                "condition": "$z_2$",
                **metrics_from_attn(attention_probs(q, k_z2), i_orig, i_tgt),
            }
        )

        k_both = swap_key_coords_in_subspace(k, i_orig, i_tgt, Vboth)
        rows.append(
            {
                "condition": "$z_1$+$z_2$",
                **metrics_from_attn(attention_probs(q, k_both), i_orig, i_tgt),
            }
        )

        # Random z1 baseline (same dimension r1)
        rand_metrics = []
        for Vr in rand_V1_list:
            k_r = swap_key_coords_in_subspace(k, i_orig, i_tgt, Vr)
            rand_metrics.append(
                metrics_from_attn(attention_probs(q, k_r), i_orig, i_tgt)
            )
        rows.append(
            {
                "condition": "Rand\n$r_1$",
                **_avg_dicts(rand_metrics),
            }
        )

        # Random z2 baseline (same dimension r2)
        rand_metrics = []
        for Vr in rand_V2_list:
            k_r = swap_key_coords_in_subspace(k, i_orig, i_tgt, Vr)
            rand_metrics.append(
                metrics_from_attn(attention_probs(q, k_r), i_orig, i_tgt)
            )
        rows.append(
            {
                "condition": "Rand\n$r_2$",
                **_avg_dicts(rand_metrics),
            }
        )

        # Random both baseline (same dimension r1+r2)
        rand_metrics = []
        for Vr in rand_Vb_list:
            k_r = swap_key_coords_in_subspace(k, i_orig, i_tgt, Vr)
            rand_metrics.append(
                metrics_from_attn(attention_probs(q, k_r), i_orig, i_tgt)
            )
        rows.append(
            {
                "condition": "Rand\n$r_1$+$r_2$",
                **_avg_dicts(rand_metrics),
            }
        )

    df = pd.DataFrame(rows)
    agg = df.groupby("condition", as_index=False).mean(numeric_only=True)

    order = [
        "Null",
        "$z_1$",
        "$z_2$",
        "$z_1$+$z_2$",
        "Rand\n$r_1$",
        "Rand\n$r_2$",
        "Rand\n$r_1$+$r_2$",
    ]
    agg["condition"] = pd.Categorical(agg["condition"], categories=order, ordered=True)
    agg = agg.sort_values("condition")
    return agg


def plot_key_swap_results(df: pd.DataFrame):
    # Configure figure sizing/typography for LaTeX subfigures.
    width_in = 3.2  # ~0.49 * 6.5in textwidth
    height_in = width_in * 0.62
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )
    # grouped bars: attention on orig vs attention on target
    df_long = pd.concat(
        [
            df[["condition", "attn_orig_mean"]]
            .rename(columns={"attn_orig_mean": "value"})
            .assign(metric="attn_on_i_orig"),
            df[["condition", "attn_tgt_mean"]]
            .rename(columns={"attn_tgt_mean": "value"})
            .assign(metric="attn_on_i_target"),
        ],
        ignore_index=True,
    )

    metrics = ["attn_on_i_orig", "attn_on_i_target"]
    metric_labels = {
        "attn_on_i_orig": r"$i_{orig}$",
        "attn_on_i_target": r"$i_{target}$",
    }

    df_pivot = (
        df_long.pivot(index="condition", columns="metric", values="value")
        .reindex(df["condition"])
        .fillna(0.0)
    )

    x = np.arange(len(df_pivot.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            df_pivot[metric].values,
            width,
            label=metric_labels.get(metric, metric),
        )

    ax.set_title("Causal Intervention on $z_1$, $z_2$ Subspaces\n$d_{head}=16$, $r_1 = 3$, $r_2 = 5$")
    ax.set_ylabel("Mean Attention")
    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index)
    ax.axvline(0.5, linestyle="--", color="0.5", linewidth=0.8)
    ax.axvline(3.5, linestyle="--", color="0.5", linewidth=0.8)
    ax.legend()
    fig.tight_layout(pad=0.2)
    plt.savefig("zxcv.pdf", bbox_inches="tight")


if __name__ == "__main__":
    dhead = 16
    N = [3]
    M = [5]

    for _N, _M in itertools.product(N, M):

        print("==============================")
        print(f"Running for N={_N}, M={_M}, dhead={dhead}")
        run_dir = Path(f"toy_model_checkpoints/dhead_{dhead}/N{_N}_M{_M}_dh{dhead}_seed11/")
        cfg = RecallConfig()
        cfg = apply_cfg_overrides(cfg, load_config(run_dir))
        set_seed(cfg.seed)
        gaussian_dgp = "gaussian" in str(run_dir).lower()

        dgp = build_dgp(cfg, gaussian_dgp)
        model = load_model(cfg, run_dir)
        print(f"Loaded run: {run_dir}")

        Delta = estimate_deltaC(cfg, dgp, model, factor="z1", mean_center=False)
        _, V_z1, _, _ = subspaces_from_deltaC(cfg, Delta)

        Delta = estimate_deltaC(cfg, dgp, model, factor="z2", mean_center=False)
        _, V_z2, _, _ = subspaces_from_deltaC(cfg, Delta)

        df = run_key_swap_experiment(
            cfg,
            dgp,
            model,
            V_z1,
            V_z2,
            n_batches=200,
            batch_size=256,
        )
        print(df)
        plot_key_swap_results(df)
