from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import repeat_kv

from constants import key_module_name, query_module_name


def pos_qr(A: Tensor) -> Tensor:
    """QR with a sign fix so output is stable up to column sign."""
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q


def orth(B: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Orthonormalize columns of B. Returns shape [d, r’] (possibly r’=0).
    """
    if B.numel() == 0:
        return B
    # remove near-zero columns
    col_norms = torch.linalg.norm(B, dim=0)
    keep = col_norms > eps
    if not torch.any(keep):
        return B[:, :0]
    B = B[:, keep]
    return pos_qr(B)


def project(x: Tensor, U: Tensor) -> Tensor:
    """
    Project x onto span(U). U assumed column-orthonormal, shape [d, r].
    x shape [..., d] -> output [..., d]
    """
    if U.numel() == 0:
        return torch.zeros_like(x)
    # coeff = x @ U  (broadcast over leading dims)
    coeff = torch.matmul(x, U)  # [..., r]
    return torch.matmul(coeff, U.transpose(-1, -2))  # [..., d]


def orthogonalize_basis(U: Tensor, U_against: Tensor) -> Tensor:
    """
    Return a basis spanning span(U) and span(U_against) removed:
      U_orth = orth((I - P_against) U)
    """
    if U.numel() == 0:
        return U
    if U_against.numel() == 0:
        return orth(U)
    # (I - P)U = U - U_against (U_against^T U)
    resid = U - U_against @ (U_against.transpose(-1, -2) @ U)
    return orth(resid)


@dataclass
class LogitsExplainedResult:
    logits_total: Tensor  # [B, S]
    logits_components: Dict[str, Tensor]  # each [B, S]
    explained_frac: Tensor  # [B] 1 - ||resid||^2 / ||total||^2


@torch.no_grad()
def extract_qk_for_head_from_cache(
    model,
    cache: dict,
    layer_idx: int,
    head_idx: int,
    query_pos: int = -1,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      q: [B, d_head] at query_pos
      K: [B, S, d_head] over all key positions
    Mirrors how you pull q/k in C_order / C_lex (incl. repeat_kv for LLaMA GQA). :contentReference[oaicite:1]{index=1}
    """
    q_name = query_module_name.format(layer_idx)
    k_name = key_module_name.format(layer_idx)

    q_all = cache[q_name][0]  # [B, n_q_heads, S, d]
    q = q_all[:, head_idx, query_pos, :]  # [B, d]

    k_all = cache[k_name][0]  # [B, n_kv_heads, S, d]
    num_groups = model.model.layers[0].self_attn.num_key_value_groups
    k_all = repeat_kv(
        k_all, num_groups
    )  # [B, n_heads, S, d] :contentReference[oaicite:2]{index=2}
    K = k_all[:, head_idx, :, :]  # [B, S, d]

    return q, K


@torch.no_grad()
def logits_explained_query_side(
    q: Tensor,  # [B, d]
    K: Tensor,  # [B, S, d]
    U_order: Tensor,  # [d, r_o]
    U_lex: Tensor,  # [d, r_l]
    *,
    method: str = "orthogonalize",  # "orthogonalize" or "residualize"
    residual_priority: str = "order",  # "order" or "lex"
    scale_by_sqrt_d: bool = False,
) -> LogitsExplainedResult:
    """
    Exact additive logits decomposition:
      logits = K q = K q_ord + K q_lex + K q_resid
    while avoiding overlap double-counting.

    method="orthogonalize":
      U_lex <- orth((I - P_order) U_lex), then take q_lex = P_lex q
      (clean, fixed partition; overlap assigned to "order".)

    method="residualize":
      residual_priority="order":
        q_ord = P_order q; q_lex = P_lex(q - q_ord)
      residual_priority="lex":
        q_lex = P_lex q; q_ord = P_order(q - q_lex)
      (avoids double-counting; depends on chosen priority.)
    """
    device = q.device
    Uo = orth(U_order.to(device))
    Ul = orth(U_lex.to(device))

    if method not in ("orthogonalize", "residualize"):
        raise ValueError(f"Unknown method={method}")
    if residual_priority not in ("order", "lex"):
        raise ValueError(f"Unknown residual_priority={residual_priority}")

    if method == "orthogonalize":
        Ul_eff = orthogonalize_basis(Ul, Uo)  # remove overlap from lex
        q_ord = project(q, Uo)
        q_lex = project(q, Ul_eff)
    else:
        if residual_priority == "order":
            q_ord = project(q, Uo)
            q_lex = project(q - q_ord, Ul)
        else:
            q_lex = project(q, Ul)
            q_ord = project(q - q_lex, Uo)

    q_resid = q - q_ord - q_lex

    # logits: [B, S]
    logits_total = torch.einsum("bsd,bd->bs", K, q)
    logits_ord = torch.einsum("bsd,bd->bs", K, q_ord)
    logits_lex = torch.einsum("bsd,bd->bs", K, q_lex)
    logits_resid = torch.einsum("bsd,bd->bs", K, q_resid)

    if scale_by_sqrt_d:
        d = q.shape[-1]
        s = d**0.5
        logits_total = logits_total / s
        logits_ord = logits_ord / s
        logits_lex = logits_lex / s
        logits_resid = logits_resid / s

    # explained fraction: 1 - ||resid||^2 / ||total||^2
    denom = torch.sum(logits_total**2, dim=-1).clamp_min(1e-12)
    explained = 1.0 - (torch.sum(logits_resid**2, dim=-1) / denom)

    return LogitsExplainedResult(
        logits_total=logits_total,
        logits_components={
            "order": logits_ord,
            "lex": logits_lex,
            "resid": logits_resid,
        },
        explained_frac=explained,
    )


def plot_logits_explained(tokens, total, order, lex, resid, head_layer, head_idx):

    start_idx = tokens.index("The")
    respond_idx = tokens.index(" Respond")
    which_idx = tokens.index(" Which")
    end_idx = [i for i, val in enumerate(tokens) if val == " Box"][-1]
    tokens = tokens[start_idx:respond_idx] + tokens[which_idx : end_idx + 1]
    total = torch.cat(
        [total[start_idx:respond_idx], total[which_idx : end_idx + 1]], dim=0
    )
    order = torch.cat(
        [order[start_idx:respond_idx], order[which_idx : end_idx + 1]], dim=0
    )
    lex = torch.cat([lex[start_idx:respond_idx], lex[which_idx : end_idx + 1]], dim=0)
    resid = torch.cat(
        [resid[start_idx:respond_idx], resid[which_idx : end_idx + 1]], dim=0
    )

    latex_rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.6,
    }

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    total = to_np(total)
    comps = {
        "Order": to_np(order),
        "Lexical": to_np(lex),
        "Residual": to_np(resid),
    }

    n = len(tokens)
    x = np.arange(n)
    # width = min(6.5, max(3.2, n * 0.28))
    # height = 2.6
    width = 10
    height = 2.5

    with plt.rc_context(latex_rc):
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

        pos_base = np.zeros(n, dtype=np.float32)
        neg_base = np.zeros(n, dtype=np.float32)
        comp_colors = {
            "Order": "#1f77b4",
            "Lexical": "#ff7f0e",
            "Residual": "#2ca02c",
        }

        for name, vals in comps.items():
            pos = np.clip(vals, 0, None)
            neg = np.clip(vals, None, 0)
            color = comp_colors[name]

            ax.bar(x, pos, bottom=pos_base, label=name, color=color, linewidth=0)
            pos_base = pos_base + pos
            ax.bar(x, neg, bottom=neg_base, color=color, linewidth=0)
            neg_base = neg_base + neg

        ax.axhline(0, linewidth=0.6, color="black")
        ax.plot(
            x,
            total,
            marker="o",
            linestyle="-",
            linewidth=0.8,
            color="black",
            markersize=2.5,
            label="Final Logit",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Logit Contribution")
        ax.set_title(f"Layer {head_layer} Head {head_idx} Attention Logits Explained")
        ax.set_xlim(-0.5, n - 0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        ax.legend(loc="upper right", frameon=False, ncol=4, columnspacing=0.8, handlelength=1.2)
        output_path = f"logits_explained_L{head_layer}_H{head_idx}.pdf"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


if __name__ == "__main__":
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hook_utils import convert_to_hooked_model, seed_all
    from dgp.build_data import build_prompt
    from dgp.schemas import SCHEMA_BOXES
    from dgp.dataset import BindingDataset
    import binding.C_order as C_order
    import binding.C_lex as C_lex
    import util_funcs

    seed = 11
    seed_all(seed)
    num_instances = 4
    max_objects = 4
    batch_size = 16
    schema = SCHEMA_BOXES

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

    order_sampler = partial(
        build_prompt,
        schema=schema,
        num_instances=num_instances,
        query_cat_idx=query_cat_id,
        answer_cat_idx=answer_cat_id,
        query_same_item=True,
        query_from_unused=False,
    )
    lexical_sampler = partial(
        build_prompt,
        schema=schema,
        num_instances=num_instances,
        query_cat_idx=query_cat_id,
        answer_cat_idx=answer_cat_id,
        query_same_item=False,
    )
    num_samples = 3000
    #num_samples = 16
    order_dataset = BindingDataset.from_sampler(order_sampler, num_samples)
    lexical_dataset = BindingDataset.from_sampler(lexical_sampler, num_samples)

    num_test_samples = 20
    test_data_sampler = partial(
        build_prompt,
        schema=schema,
        num_instances=num_instances,
        query_cat_idx=query_cat_id,
        answer_cat_idx=answer_cat_id,
        query_from_unused=True,
    )
    test_dataset = BindingDataset.from_sampler(test_data_sampler, num_test_samples)[
        :num_test_samples
    ]["input"]
    plot_idx = [test_dataset[idx]["queryIndex"] for idx in range(len(test_dataset))].index(1)


    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    convert_to_hooked_model(model)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # %%

    filtered_heads = [
        (16, 1),
    ]
    hook_layers = set([layer for (layer, head) in filtered_heads])
    record_module_names = [
        key_module_name.format(hook_layer) for hook_layer in hook_layers
    ] + [query_module_name.format(hook_layer) for hook_layer in hook_layers]

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

    for head in filtered_heads:

        print(f"=== Head L{head[0]}H{head[1]} ===")
        hook_layer = head[0]
        head_idx = head[1]

        U_order, S_order, Vh_order, rank_order = util_funcs.get_qk_subspace(
            C_order_pos[head], C_order_neg[head], thresh=0.99
        )
        U_lex, S_lex, Vh_lex, rank_lex = util_funcs.get_qk_subspace(
            C_lex_pos[head], C_lex_neg[head], thresh=0.99
        )
        record_names = [
            key_module_name.format(hook_layer),
            query_module_name.format(hook_layer),
        ]
        batch = test_dataset[plot_idx: plot_idx + 1]
        prompts = util_funcs.format_prompts(
            tokenizer, [sample["raw_input"] for sample in batch]
        )
        cache, _, input_ids, attention_mask = util_funcs.run_forward_pass(
            model, tokenizer, record_names, prompts
        )

        # _q: [batch, d_head]
        # _K: [batch, seq, d_head]
        _q, _K = extract_qk_for_head_from_cache(
            model, cache, layer_idx=hook_layer, head_idx=head_idx, query_pos=-1
        )

        # res.logits_total: [batch, seq]
        # res.logits_components: {
        #     "order": [batch, seq],
        #     "lex": [batch, seq],
        #     "resid": [batch, seq],
        # }
        # res.explained_frac: [batch]

        res = logits_explained_query_side(
            q=_q,
            K=_K,
            U_order=U_order[:, :rank_order],
            U_lex=U_lex[:, :rank_lex],
            # method="orthogonalize",  # or "residualize"
            method="residualize",
            residual_priority="order",
            scale_by_sqrt_d=False,
        )

        _tokens = util_funcs.to_str_tokens(tokenizer, prompts[0])
        _total = res.logits_total[0]
        _order = res.logits_components["order"][0]
        _lex = res.logits_components["lex"][0]
        _resid = res.logits_components["resid"][0]
        plot_logits_explained(
            _tokens,
            _total,
            _order,
            _lex,
            _resid,
            head_layer=hook_layer,
            head_idx=head_idx,
        )
    print("Done.")
