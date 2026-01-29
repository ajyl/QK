import math
from typing import Tuple, Literal
import torch
from dataclasses import dataclass


def sample_z(shape, device) -> torch.Tensor:
    # Uniform in {-1, +1}^shape
    return (torch.randint(0, 2, shape, device=device, dtype=torch.int64) * 2 - 1).to(
        torch.float32
    )


class LatentRecallDGP:
    """
    Generates:
      x_ctx[i] = A1 u1[i] + A2 u2[i] + e(y[i]) + eps
      x_q      = B1 u1*   + B2 u2*            + eps
    where u1 in {-1,+1}^N and u2 in {-1,+1}^M
    Query comes from a DIFFERENT encoder (B1,B2) to emulate disjoint query/context vocabularies.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        d, N, M, P = cfg.d_model, cfg.N, cfg.M, cfg.P
        device = cfg.device

        # Fixed random "token embedding" encoders
        self.A1 = torch.randn(d, N, device=device) / math.sqrt(N)
        self.A2 = torch.randn(d, M, device=device) / math.sqrt(M)
        self.B1 = torch.randn(d, N, device=device) / math.sqrt(N)
        self.B2 = torch.randn(d, M, device=device) / math.sqrt(M)

        # Payload embedding (fixed)
        self.Ey = torch.randn(P, cfg.d_model, device=device) / math.sqrt(cfg.d_model)

    @torch.no_grad()
    def sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
          x_seq: [B, C+1, d_model]  (context then query)
          y_target: [B] in [0,P)
          info: dict with latents for analysis
        """
        cfg = self.cfg
        device = cfg.device
        C, N, M, P = cfg.C, cfg.N, cfg.M, cfg.P

        # Context latents
        u1 = sample_z((B, C, N), device)
        u2 = sample_z((B, C, M), device)

        # Payload labels per context position
        if cfg.unique_labels:
            if P < C:
                raise ValueError(
                    f"unique_labels=True requires P >= C (got P={P}, C={C})."
                )
            y = torch.stack(
                [torch.randperm(P, device=device)[:C] for _ in range(B)], dim=0
            )
        else:
            y = torch.randint(0, P, (B, C), device=device, dtype=torch.int64)
        # y = self.label_from_latents(u1, u2)

        # Choose target index i* (ensure unique by construction)
        i_star = torch.randint(0, C, (B,), device=device, dtype=torch.int64)

        # Query latents = latents of target position (optionally corrupted)
        u1_star = u1[torch.arange(B, device=device), i_star].clone()
        u2_star = u2[torch.arange(B, device=device), i_star].clone()

        # Build context embeddings
        eps_ctx = torch.randn(B, C, cfg.d_model, device=device) * cfg.noise_std
        ey = self.Ey[y]  # [B, C, P]
        x_ctx = (
            (u1 @ self.A1.T) + (u2 @ self.A2.T) + ey + eps_ctx  # [B, C, d]  # [B, C, d]
        )

        # Build query embedding (no payload)
        eps_q = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        x_q = (u1_star @ self.B1.T) + (u2_star @ self.B2.T) + eps_q  # [B, d]

        # Sequence = context then query
        x_seq = torch.cat([x_ctx, x_q[:, None, :]], dim=1)  # [B, C+1, d]

        # Target payload is y at i*
        y_target = y[torch.arange(B, device=device), i_star]  # [B]

        info = {
            "u1": u1,
            "u2": u2,
            "y": y,
            "i_star": i_star,
            "u1_star": u1_star,
            "u2_star": u2_star,
        }
        return x_seq, y_target, info

    @torch.no_grad()
    def sample_contrast_batch(
        self, B: int, factor: Literal["z1", "z2", "both", "none"] = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        device = cfg.device
        N, M, P = cfg.N, cfg.M, cfg.P

        # Sample query latents
        u1_star = sample_z((B, N), device)
        u2_star = sample_z((B, M), device)

        # Query embedding
        eps_q = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        x_q = (u1_star @ self.B1.T) + (u2_star @ self.B2.T) + eps_q  # [B, d]

        # Payload nuisance (kept realistic)
        y_pos = torch.randint(0, P, (B,), device=device, dtype=torch.int64)
        y_neg = torch.randint(0, P, (B,), device=device, dtype=torch.int64)
        ey_pos = self.Ey[y_pos]
        ey_neg = self.Ey[y_neg]

        eps_kp = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        eps_kn = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std

        if factor == "both":
            # + shares both factors; - shares neither
            u1_pos, u2_pos = u1_star, u2_star
            u1_neg, u2_neg = sample_z((B, N), device), sample_z((B, M), device)

        elif factor == "z1":
            # HOLD Z2 FIXED across pos/neg; only Z1 differs
            u2_fixed = sample_z((B, M), device)
            u1_pos, u2_pos = u1_star, u2_fixed
            u1_neg, u2_neg = sample_z((B, N), device), u2_fixed

        elif factor == "z2":
            # HOLD Z1 FIXED across pos/neg; only Z2 differs
            u1_fixed = sample_z((B, N), device)
            u1_pos, u2_pos = u1_fixed, u2_star
            u1_neg, u2_neg = u1_fixed, sample_z((B, M), device)
        else:
            raise ValueError(f"Unknown factor: {factor}")

        x_k_pos = (u1_pos @ self.A1.T) + (u2_pos @ self.A2.T) + ey_pos + eps_kp
        x_k_neg = (u1_neg @ self.A1.T) + (u2_neg @ self.A2.T) + ey_neg + eps_kn

        return x_q, x_k_pos, x_k_neg


@dataclass
class Config:
    # Latent intrinsic dims (Option 1)
    r1: int = 3
    r2: int = 3

    # Sequence
    C: int = 12

    # Payload classes (keep if you still want classification)
    P: int = 32

    # Model dims
    d_model: int = 64
    d_head: int = 32

    # DGP noise
    noise_std: float = 0.10

    # Optional: corruption of query latents (Gaussian noise in latent space)
    latent_noise_std: float = 0.00  # set e.g. 0.05 to make "approximate match"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GaussianTwoFactorRecallDGP:
    """
    Continuous-latent version:
      s1 in R^{r1}, s2 in R^{r2}, both standard normal.
      Context token: x_i = A1 s1_i + A2 s2_i + e(y_i) + eps
      Query token:   x_q = B1 s1*  + B2 s2*            + eps
    Query uses separate encoder (B1,B2) to emulate disjoint query/context vocabularies.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        device = cfg.device
        d, r1, r2, P = cfg.d_model, cfg.r1, cfg.r2, cfg.P

        # Fixed random encoders (token-embedding generators)
        self.A1 = torch.randn(d, r1, device=device) / math.sqrt(r1)
        self.A2 = torch.randn(d, r2, device=device) / math.sqrt(r2)
        self.B1 = torch.randn(d, r1, device=device) / math.sqrt(r1)
        self.B2 = torch.randn(d, r2, device=device) / math.sqrt(r2)

        # Payload embedding
        self.Ey = torch.randn(P, cfg.d_model, device=device) / math.sqrt(cfg.d_model)

    @torch.no_grad()
    def sample_batch(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
          x_seq: [B, C+1, d_model]  (context then query)
          y_target: [B] in [0,P)
          info: dict with latents for analysis
        """
        cfg = self.cfg
        device = cfg.device
        C, r1, r2, P = cfg.C, cfg.r1, cfg.r2, cfg.P

        # Context latents
        s1 = torch.randn(B, C, r1, device=device)
        s2 = torch.randn(B, C, r2, device=device)

        # Payload labels per context position
        y = torch.randint(0, P, (B, C), device=device, dtype=torch.int64)

        # Choose target index i*
        i_star = torch.randint(0, C, (B,), device=device, dtype=torch.int64)

        # Query latents = latents of target position (+ optional latent noise)
        s1_star = s1[torch.arange(B, device=device), i_star].clone()
        s2_star = s2[torch.arange(B, device=device), i_star].clone()
        if cfg.latent_noise_std > 0:
            s1_star = s1_star + torch.randn_like(s1_star) * cfg.latent_noise_std
            s2_star = s2_star + torch.randn_like(s2_star) * cfg.latent_noise_std

        # Build context embeddings
        eps_ctx = torch.randn(B, C, cfg.d_model, device=device) * cfg.noise_std
        ey = self.Ey[y]  # [B, C, P]
        x_ctx = (
            (s1 @ self.A1.T)  # [B, C, d]
            + (s2 @ self.A2.T)  # [B, C, d]
            + ey
            + eps_ctx
        )

        # Build query embedding (no payload)
        eps_q = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        x_q = (s1_star @ self.B1.T) + (s2_star @ self.B2.T) + eps_q  # [B, d]

        # Sequence = context then query
        x_seq = torch.cat([x_ctx, x_q[:, None, :]], dim=1)  # [B, C+1, d]

        # Target payload is y at i*
        y_target = y[torch.arange(B, device=device), i_star]  # [B]

        info = {
            "s1": s1,
            "s2": s2,
            "y": y,
            "i_star": i_star,
            "s1_star": s1_star,
            "s2_star": s2_star,
        }
        return x_seq, y_target, info

    @torch.no_grad()
    def sample_contrast_batch(
        self,
        B: int,
        factor: Literal["z1", "z2", "both"] = "both",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_q, x_k_pos, x_k_neg) in model-input space.
        We'll later map them through model W_Q/W_K to estimate ΔC in head space.

        IMPORTANT (for rank isolation):
          - "z1": hold s2 fixed across pos/neg; only s1 differs
          - "z2": hold s1 fixed across pos/neg; only s2 differs
          - "both": pos shares both; neg shares neither
        """
        cfg = self.cfg
        device = cfg.device
        r1, r2, P = cfg.r1, cfg.r2, cfg.P

        # Query latents
        s1_star = torch.randn(B, r1, device=device)
        s2_star = torch.randn(B, r2, device=device)
        if cfg.latent_noise_std > 0:
            s1_star = s1_star + torch.randn_like(s1_star) * cfg.latent_noise_std
            s2_star = s2_star + torch.randn_like(s2_star) * cfg.latent_noise_std

        # Query embedding
        eps_q = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        x_q = (s1_star @ self.B1.T) + (s2_star @ self.B2.T) + eps_q  # [B, d]

        # Payload nuisance (realistic; keys should learn to ignore it)
        y_pos = torch.randint(0, P, (B,), device=device, dtype=torch.int64)
        y_neg = torch.randint(0, P, (B,), device=device, dtype=torch.int64)
        ey_pos = self.Ey[y_pos]  # [B,P]
        ey_neg = self.Ey[y_neg]  # [B,P]

        eps_kp = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std
        eps_kn = torch.randn(B, cfg.d_model, device=device) * cfg.noise_std

        if factor == "both":
            s1_pos, s2_pos = s1_star, s2_star
            s1_neg, s2_neg = torch.randn(B, r1, device=device), torch.randn(
                B, r2, device=device
            )

        elif factor == "z1":
            # Hold s2 fixed so it cancels in ΔC
            s2_fixed = torch.randn(B, r2, device=device)
            s1_pos, s2_pos = s1_star, s2_fixed
            s1_neg, s2_neg = torch.randn(B, r1, device=device), s2_fixed

        elif factor == "z2":
            # Hold s1 fixed so it cancels in ΔC
            s1_fixed = torch.randn(B, r1, device=device)
            s1_pos, s2_pos = s1_fixed, s2_star
            s1_neg, s2_neg = s1_fixed, torch.randn(B, r2, device=device)

        else:
            raise ValueError(f"Unknown factor: {factor}")

        # Context-like embeddings used as keys
        x_k_pos = (
            (s1_pos @ self.A1.T) + (s2_pos @ self.A2.T) + ey_pos + eps_kp
        )
        x_k_neg = (
            (s1_neg @ self.A1.T) + (s2_neg @ self.A2.T) + ey_neg + eps_kn
        )

        return x_q, x_k_pos, x_k_neg
