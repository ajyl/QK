import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv

from categories.filter_head_utils import (
    build_filter_dataset,
    collect_item_timesteps_from_input_ids,
)
from util_funcs import format_prompts, run_forward_pass
from constants import key_module_name, query_module_name


def index_query_key_vecs(
    cache,
    layer_idx: int,
    key_token_idxs: torch.Tensor,  # [B, M]
    num_key_value_groups: int,
):
    """
    Extract per-head query vectors at the last token and key vectors at item positions.

    q_vecs: [B, H, d_head]
    k_vecs: [B, H, M, d_head]
    """
    q_all = cache[query_module_name.format(layer_idx)][0].cpu()
    q_vecs = q_all[:, :, -1, :]

    k_all = cache[key_module_name.format(layer_idx)][0].cpu()
    k_all = repeat_kv(k_all, num_key_value_groups)

    B, H, S, D = k_all.shape
    _B, M = key_token_idxs.shape
    if _B != B:
        raise ValueError(f"Batch size mismatch: key_token_idxs={_B}, k_all={B}")

    pos_idx = key_token_idxs[:, None].expand(B, H, M).to(k_all.device)
    k_vecs = k_all.gather(
        dim=2,
        index=pos_idx.unsqueeze(-1).expand(B, H, M, D),
    )
    return q_vecs, k_vecs


def _build_C_filter_heads(
    model,
    tokenizer,
    record_module_names,
    samples,
    filtered_heads,
    batch_size: int,
    categories=None,
):
    """
    C_pos[(layer, head)] -> [C, d_head, d_head]
    C_neg[(layer, head)] -> [C, d_head, d_head]
    counts[(layer, head)] -> [C]
    """
    if categories is None:
        categories = sorted({x["predicate"] for x in samples})
    cat2idx = {c: i for i, c in enumerate(categories)}

    d_head = model.config.head_dim
    num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups

    C_pos = {
        _head: torch.zeros(len(categories), d_head, d_head) for _head in filtered_heads
    }
    C_neg = {
        _head: torch.zeros(len(categories), d_head, d_head) for _head in filtered_heads
    }
    counts = {
        _head: torch.zeros(len(categories), dtype=torch.long)
        for _head in filtered_heads
    }

    for batch_start in tqdm(range(0, len(samples), batch_size)):
        batch = samples[batch_start : batch_start + batch_size]
        if not batch:
            continue

        prompts = format_prompts(tokenizer, [x["raw_input"] for x in batch])
        cache, _, input_ids, _ = run_forward_pass(
            model, tokenizer, record_module_names, prompts
        )

        item_pos = collect_item_timesteps_from_input_ids(
            tokenizer, input_ids, [x["items"] for x in batch], prefer_last_token=True
        )
        item_pos = torch.tensor(item_pos, device=model.device)

        B, M = item_pos.shape
        pos_mask = torch.zeros(B, M, device=model.device, dtype=torch.bool)
        neg_mask = torch.zeros(B, M, device=model.device, dtype=torch.bool)
        cat_idx = torch.zeros(B, device=model.device, dtype=torch.long)
        for b, x in enumerate(batch):
            pos_mask[b, x["pos_item_idxs"]] = True
            neg_mask[b, x["neg_item_idxs"]] = True
            cat_idx[b] = cat2idx[x["predicate"]]

        layers = sorted(list({layer for (layer, _) in filtered_heads}))
        layer_qk_cache = {}
        for layer_idx in layers:
            q_vecs, k_vecs = index_query_key_vecs(
                cache,
                layer_idx,
                item_pos,
                num_key_value_groups,
            )
            layer_qk_cache[layer_idx] = (q_vecs.cpu(), k_vecs.cpu())

        for _head in filtered_heads:
            layer_idx, head_idx = _head
            q_vecs, k_vecs = layer_qk_cache[layer_idx]

            q_h = q_vecs[:, head_idx, :].to(model.device)  # [B, d_head]
            k_h = k_vecs[:, head_idx, :].to(model.device)  # [B, M, d_head]

            for b in range(B):
                c_idx = cat_idx[b].item()
                k_pos = k_h[b, pos_mask[b]]
                k_neg = k_h[b, neg_mask[b]]

                if k_pos.numel() > 0:
                    q_pos = q_h[b][None, :].expand(k_pos.shape[0], -1)
                    C_pos[_head][c_idx] += torch.einsum("nq,nk->qk", q_pos, k_pos).cpu()
                if k_neg.numel() > 0:
                    q_neg = q_h[b][None, :].expand(k_neg.shape[0], -1)
                    C_neg[_head][c_idx] += torch.einsum("nq,nk->qk", q_neg, k_neg).cpu()

                counts[_head][c_idx] += 1

    return C_pos, C_neg, counts, categories


def build_C_filter_heads(
    model,
    tokenizer,
    record_module_names,
    samples,
    filtered_heads,
    batch_size: int,
    categories=None,
):
    """
    Build per-category C_pos, C_neg, and Delta_C for each head.

    Returns:
        C_filter_pos: dict[(layer, head)] -> [C, d_head, d_head]
        C_filter_neg: dict[(layer, head)] -> [C, d_head, d_head]
        Delta_C:      dict[(layer, head)] -> [C, d_head, d_head]
        counts:       dict[(layer, head)] -> [C]
        categories:   list[str]
    """
    C_pos, C_neg, counts, categories = _build_C_filter_heads(
        model,
        tokenizer,
        record_module_names,
        samples,
        filtered_heads,
        batch_size,
        categories,
    )

    Delta_C = {}
    for _head in filtered_heads:
        for c in range(len(categories)):
            if counts[_head][c] > 0:
                C_pos[_head][c] /= counts[_head][c].float()
                C_neg[_head][c] /= counts[_head][c].float()
        Delta_C[_head] = C_pos[_head] - C_neg[_head]

    return C_pos, C_neg, Delta_C, counts, categories


