from __future__ import annotations

import argparse
import json
import copy
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgp import LatentRecallDGP, GaussianTwoFactorRecallDGP, Config as GaussianDGPConfig
from util_funcs import (
    set_seed,
    effective_rank_from_svals,
)


@dataclass
class Config:
    # Latent sizes
    N: int = 3  # dim of Z1
    M: int = 5  # dim of Z2

    # Sequence
    C: int = 16  # number of context tokens (sequence length = C + 1 including query)

    # Payload classes
    P: int = 10
    unique_labels: bool = False

    # Model dims
    d_model: int = 32
    d_head: int = 16

    # DGP noise
    noise_std: float = 0.01

    # Training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    steps: int = 50000
    lr: float = 1e-4
    weight_decay: float = 1e-2
    log_every: int = 200
    eval_C_every: int = 2000
    val_batches: int = 20
    val_batch_size: int = 512
    val_every: int = 200
    patience: int = 5
    min_delta: float = 1e-4

    # ΔC estimation
    deltaC_batches: int = 80
    deltaC_batch_size: int = 512
    energy_threshold: float = 0.99

    seed: int = 11


@torch.no_grad()
def svd_report(
    name: str, M: torch.Tensor, energy: float = 0.99
) -> Tuple[torch.Tensor, int]:
    # M: [dh,dh]
    s = torch.linalg.svdvals(M)
    r_eff = effective_rank_from_svals(s, energy=energy)
    top = s[: min(10, s.numel())].detach().cpu()
    print(
        f"{name}: ||ΔC||_F={M.norm().item():.4f} | top svals={top.numpy()} | eff_rank@{energy:.2f}={r_eff}"
    )
    return s, r_eff



class RecallDGP(Protocol):
    def sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, dict]: ...

    def sample_contrast_batch(
        self, B: int, factor: Literal["z1", "z2", "both"] = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class Attention(nn.Module):
    """
    Attention-only retrieval:
      - query is last token
      - keys/values are context tokens
      - output at query is weighted sum of values, then classify payload
    """

    def __init__(self, d_model: int, d_head: int, P: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.P = P

        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

        self.out = nn.Linear(d_head, P, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, C+1, d_model]
        returns logits: [B, P]
        """
        B, L, d = x_seq.shape
        assert d == self.d_model
        C = L - 1

        x_ctx = x_seq[:, :C, :]  # [B,C,d]
        x_q = x_seq[:, C, :]  # [B,d]

        q = self.W_Q(x_q)  # [B,dh]
        k = self.W_K(x_ctx)  # [B,C,dh]
        v = self.W_V(x_ctx)  # [B,C,dh]

        # attention scores: [B,C]
        scores = torch.einsum("bd,bcd->bc", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)

        # weighted sum: [B,dh]
        o = torch.einsum("bc,bcd->bd", attn, v)

        logits = self.out(o)  # [B,P]
        return logits

    @torch.no_grad()
    def get_qk(
        self, x_q: torch.Tensor, x_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience for ΔC estimation.
        x_q: [B,d_model]
        x_k: [B,d_model]
        returns q: [B,dh], k: [B,dh]
        """
        return self.W_Q(x_q), self.W_K(x_k)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train(cfg: Config, dgp: RecallDGP) -> tuple[Attention, float]:
    model = Attention(cfg.d_model, cfg.d_head, cfg.P).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Fixed validation set for early stopping
    val_data = [dgp.sample_batch(cfg.val_batch_size) for _ in range(cfg.val_batches)]
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.patience

    def eval_val_loss() -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x_seq, y_tgt, _ in val_data:
                logits = model(x_seq)
                loss = F.cross_entropy(logits, y_tgt)
                total += loss.item() * x_seq.shape[0]
                n += x_seq.shape[0]
        model.train()
        return total / max(1, n)

    model.train()
    for step in range(1, cfg.steps + 1):
        # x_seq: [batch, context (C + 1), d_model]
        # y_tgt: [batch] (int)
        # meta["u1"]: [batch, C, N]
        # meta["u2"]: [batch, C, M]
        # meta["y"]: [batch, C]
        # meta["i_star"]: [batch] (int)
        # meta["u1_star"]: [batch, N]
        # meta["u2_star"]: [batch, M]
        x_seq, y_tgt, meta = dgp.sample_batch(cfg.batch_size)

        # logits: [batch, num_classes (P)]
        logits = model(x_seq)
        loss = F.cross_entropy(logits, y_tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = accuracy(logits, y_tgt)
            print(f"step {step:5d} | loss {loss.item():.4f} | acc {acc*100:5.1f}%")

        if step % cfg.val_every == 0 or step == 1:
            val_loss = eval_val_loss()
            improved = val_loss < (best_val - cfg.min_delta)
            if improved:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_left = cfg.patience
            else:
                patience_left -= 1
            print(
                f"step {step:5d} | val_loss {val_loss:.4f} | "
                f"patience {patience_left}/{cfg.patience}"
            )
            if patience_left <= 0:
                print("Early stopping: validation loss did not improve.")
                break

        if step % cfg.eval_C_every == 0 and step > 0:

            ## Factor-isolated ΔC (decomposability demo)
            Delta_z1 = estimate_deltaC(cfg, dgp, model, factor="z1", mean_center=False)
            Delta_z2 = estimate_deltaC(cfg, dgp, model, factor="z2", mean_center=False)
            s1, r1 = svd_report("ΔC_z1", Delta_z1, energy=cfg.energy_threshold)
            s2, r2 = svd_report("ΔC_z2", Delta_z2, energy=cfg.energy_threshold)

            print(
                f"Expected (roughly): rank(ΔC_z1)≈N={cfg.N}, rank(ΔC_z2)≈M={cfg.M}, rank(ΔC_both)≈N+M={cfg.N+cfg.M}"
            )
            print(f"Observed eff ranks: z1={r1}, z2={r2}")
            print("")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def estimate_deltaC(
    cfg: Config,
    dgp: RecallDGP,
    model: Attention,
    factor: Literal["z1", "z2", "both"] = "both",
    mean_center: bool = False,
) -> torch.Tensor:
    """
    Estimates ΔC in head space:
      ΔC = E[q k_pos^T] - E[q k_neg^T]
    q,k are the model's projected vectors (after W_Q/W_K).

    mean_center: if True, estimate covariance-style moments:
      E[(q-μq)(k-μk)^T]
    """
    device = cfg.device
    dh = cfg.d_head

    sum_pos = torch.zeros(dh, dh, device=device)
    sum_neg = torch.zeros(dh, dh, device=device)

    # For optional centering
    if mean_center:
        sum_q = torch.zeros(dh, device=device)
        sum_kp = torch.zeros(dh, device=device)
        sum_kn = torch.zeros(dh, device=device)

    n_total = 0

    model.eval()
    for _ in range(cfg.deltaC_batches):
        x_q, x_kp, x_kn = dgp.sample_contrast_batch(
            cfg.deltaC_batch_size, factor=factor
        )
        q, kp = model.get_qk(x_q, x_kp)
        _, kn = model.get_qk(x_q, x_kn)

        B = q.shape[0]
        n_total += B

        if mean_center:
            sum_q += q.sum(dim=0)
            sum_kp += kp.sum(dim=0)
            sum_kn += kn.sum(dim=0)

        sum_pos += q.T @ kp
        sum_neg += q.T @ kn

    C_pos = sum_pos / n_total
    C_neg = sum_neg / n_total

    if mean_center:
        mu_q = sum_q / n_total
        mu_kp = sum_kp / n_total
        mu_kn = sum_kn / n_total
        C_pos = C_pos - mu_q[:, None] @ mu_kp[None, :]
        C_neg = C_neg - mu_q[:, None] @ mu_kn[None, :]

    return C_pos - C_neg


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(
        description="Train the recall model and report ΔC diagnostics."
    )
    parser.add_argument(
        "--N",
        type=int,
    )
    parser.add_argument(
        "--M",
        type=int,
    )
    parser.add_argument(
        "--d-head",
        type=int,
    )
    parser.add_argument(
        "--gaussian-dgp",
        action="store_true",
        help="Use GaussianTwoFactorRecallDGP instead of LatentRecallDGP.",
    )
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    cfg.N = args.N
    cfg.M = args.M
    cfg.d_head = args.d_head

    if args.gaussian_dgp:
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
        dgp = GaussianTwoFactorRecallDGP(gcfg)
    else:
        dgp = LatentRecallDGP(cfg)
    model, _ = train(cfg, dgp)

    Delta_z1 = estimate_deltaC(cfg, dgp, model, factor="z1", mean_center=False)
    Delta_z2 = estimate_deltaC(cfg, dgp, model, factor="z2", mean_center=False)

    print(
        f"Expected (roughly): rank(ΔC_z1)≈N={cfg.N}, rank(ΔC_z2)≈M={cfg.M}, rank(ΔC_both)≈N+M={cfg.N+cfg.M}"
    )

    if args.gaussian_dgp:
        out_dir = Path(f"toy_model_checkpoints/dhead_{cfg.d_head}_gaussian")
        run_dir = out_dir / f"gaussian_N{cfg.N}_M{cfg.M}_dh{cfg.d_head}_seed{cfg.seed}"
    else:
        out_dir = Path(f"toy_model_checkpoints/dhead_{cfg.d_head}")
        run_dir = out_dir / f"N{cfg.N}_M{cfg.M}_dh{cfg.d_head}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", asdict(cfg))
    torch.save(model.state_dict(), run_dir / "model.pt")
    torch.save(Delta_z1, run_dir / "delta_z1.pt")
    torch.save(Delta_z2, run_dir / "delta_z2.pt")


if __name__ == "__main__":
    main()
