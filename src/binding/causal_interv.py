import os
from functools import partial

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv
from tqdm import tqdm

from hook_utils import convert_to_hooked_model, seed_all
from dgp.build_data import build_prompt
from dgp.schemas import SCHEMA_BOXES
from dgp.dataset import BindingDataset
import binding.C_order as C_order
import binding.C_lex as C_lex
import util_funcs

from constants import (
    BASE_DIR,
    key_module_name,
    query_module_name,
    attn_module_name,
    qk_module_name,
)


# %%


def edit_qk(q_vecs, k_orig, k_target, U_r, V_r):
    coeff_orig = k_orig @ V_r
    coeff_target = k_target @ V_r

    orig_edited = k_orig + (coeff_target - coeff_orig) @ V_r.T
    target_edited = k_target + (coeff_orig - coeff_target) @ V_r.T

    new_logits_orig = (q_vecs * orig_edited).sum(dim=-1)
    new_logits_target = (q_vecs * target_edited).sum(dim=-1)
    return new_logits_orig, new_logits_target


def edit_qk_both(q_vecs, k_orig, k_target, U_r, V_r, U_r_second, V_r_second):
    coeff_orig = k_orig @ V_r
    coeff_target = k_target @ V_r
    # delta = (coeff_target - coeff_orig) @ V_r.T

    coeff_orig_second = k_orig @ V_r_second
    coeff_target_second = k_target @ V_r_second
    # delta_second = (coeff_target_second - coeff_orig_second) @ V_r_second.T

    orig_edited = (
        k_orig
        + (coeff_target - coeff_orig) @ V_r.T
        + (coeff_target_second - coeff_orig_second) @ V_r_second.T
    )
    target_edited = (
        k_target
        + (coeff_orig - coeff_target) @ V_r.T
        + (coeff_orig_second - coeff_target_second) @ V_r_second.T
    )

    new_logits_orig = (q_vecs * orig_edited).sum(dim=-1)
    new_logits_target = (q_vecs * target_edited).sum(dim=-1)
    return new_logits_orig, new_logits_target


def pos_qr(A):
    Q, R = torch.linalg.qr(A)
    signs = torch.diag(torch.sign(torch.diag(R)))
    Q = Q @ signs
    return Q


def build_random_key_basis(key_basis, rng):
    dim, rank = key_basis.shape
    rand = rng.standard_normal(size=(dim, rank))
    rand_t = torch.from_numpy(rand).to(dtype=key_basis.dtype)
    return pos_qr(rand_t)



@torch.no_grad()
def run_causal_interventions_for_head(
    model,
    tokenizer,
    dataset,
    schema,
    hook_layer,
    head_idx,
    U,
    Vh,
    rank,
    num_instances,
    batch_size,
    second=None,
):
    """
    For each example:
      1) Identify answer box token and a target box token.
      2) Baseline run: record attn and Q,K.
      3) Offline: edit target key in order subspace, compute new qk logit.
      4) Second run: hook into qk logits to inject edited logit.
      5) Measure change in attention mass from answer->target.

    Returns dict of aggregate metrics.
    """
    util_funcs.remove_all_hooks(model)

    key_matcher = schema.matchers[1]  # Box
    query_matcher = schema.matchers[0]  # Objects

    key_name = key_module_name.format(hook_layer)
    query_name = query_module_name.format(hook_layer)
    attn_name = attn_module_name.format(hook_layer)

    U_r = U[:, :rank].to(model.device)  # [d_head, r]
    V_full = Vh.T  # [d_head, d_head]
    V_r = V_full[:, :rank].to(model.device)  # [d_head, r]

    all_delta_p = []
    all_flip = []
    all_orig_a = []
    all_orig_a_target = []
    all_interv_a = []
    all_interv_a_target = []
    all_attn_orig = []
    all_attn_interv = []

    named_modules = dict(model.named_modules())
    qk_name = qk_module_name.format(hook_layer)
    if qk_name not in named_modules:
        raise ValueError(
            f"{qk_name} not found in model.named_modules(). "
            "You need a HookPoint there (e.g. in convert_to_hooked_model)."
        )

    num_batches_run = 0

    for batch_idx in range(0, len(dataset), batch_size):
        batch = dataset[batch_idx : batch_idx + batch_size]
        if len(batch) == 0:
            continue

        prompts = util_funcs.format_prompts(
            tokenizer, [sample["raw_input"] for sample in batch]
        )
        key_timesteps, _ = C_order.collect_key_query_timesteps(
            prompts, tokenizer, key_matcher, query_matcher
        )
        ans_indices = [sample["queryIndex"] for sample in batch]
        # choose a random different box as the “target”
        target_indices = []
        for b_idx, sample in enumerate(batch):
            ans_idx = sample["queryIndex"]
            possible_targets = [i for i in range(num_instances) if i != ans_idx]
            target_idx = np.random.choice(possible_targets)
            target_indices.append(target_idx)

        # token-level positions
        orig_token_idxs = torch.tensor(
            [key_timesteps[b_idx][ans_indices[b_idx]] for b_idx in range(len(batch))],
            device=model.device,
        )
        target_token_idxs = torch.tensor(
            [
                key_timesteps[b_idx][target_indices[b_idx]]
                for b_idx in range(len(batch))
            ],
            device=model.device,
        )

        record_names = [key_name, query_name, attn_name]
        cache, _, input_ids, attention_mask = util_funcs.run_forward_pass(
            model, tokenizer, record_names, prompts
        )

        q_states = cache[query_name][0]  # [B,H,seq,D]
        k_states = cache[key_name][0]  # [B,H_kv,seq,D]
        k_states = repeat_kv(
            k_states, model.model.layers[0].self_attn.num_key_value_groups
        )  # [B,H,seq,D]

        attn_base = cache[attn_name][0][:, head_idx, -1, :]  # [B, seq]

        B, H, S, D = k_states.shape
        assert head_idx < H

        b_idx = torch.arange(B, device=model.device)

        q_vecs = q_states[:, head_idx, -1, :]  # [B,D]
        k_states = k_states[:, head_idx, :, :]  # [B,seq,D]
        k_orig = k_states[b_idx, orig_token_idxs, :]  # [B,D]
        k_target = k_states[b_idx, target_token_idxs, :]  # [B,D]

        if second is None:
            l_orig_new, l_target_new = edit_qk(q_vecs, k_orig, k_target, U_r, V_r)
        else:
            rank_second = second[2]
            U_second = second[0][:, :rank_second].to(model.device)
            Vh_second = second[1].T[:, :rank_second].to(model.device)
            l_orig_new, l_target_new = edit_qk_both(
                q_vecs, k_orig, k_target, U_r, V_r, U_second, Vh_second
            )

        attn_int_cache = []

        def qk_hook(_module, _inputs, output):
            # output: [B, H, q_len, kv_len]
            qk = output
            B_local, H_local, Q_local, K_local = qk.shape
            assert B_local == B
            assert H_local > head_idx
            q_pos = Q_local - 1  # assume final position is the query
            b_local = torch.arange(B_local, device=qk.device)
            # overwrite target logit for this head at the query position
            qk[b_local, head_idx, q_pos, orig_token_idxs] = l_orig_new.to(qk.device)
            qk[b_local, head_idx, q_pos, target_token_idxs] = l_target_new.to(qk.device)
            return qk

        def attn_hook(_module, _inputs, output):
            # output: [B,H,q_len,kv_len]
            attn_int_cache.append(output.detach().cpu())
            return output

        qk_module = named_modules[qk_name]
        attn_module = named_modules[attn_name]

        h1 = qk_module.register_forward_hook(qk_hook)
        h2 = attn_module.register_forward_hook(attn_hook)

        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        h1.remove()
        h2.remove()

        assert len(attn_int_cache) == 1
        attn_int = attn_int_cache[0][:, head_idx, -1, :]  # [B, seq]

        # === 4) Metrics: Δp and flip rate ===
        attn_base_np = attn_base.cpu().numpy()
        attn_int_np = attn_int.cpu().numpy()

        for b in range(B):
            ans_tok = orig_token_idxs[b].item()
            target_tok = target_token_idxs[b].item()

            alpha = attn_base_np[b]
            alpha_tilde = attn_int_np[b]

            a_ans = alpha[ans_tok]
            a_target = alpha[target_tok]
            a_ans_t = alpha_tilde[ans_tok]
            a_target_t = alpha_tilde[target_tok]

            denom0 = a_ans + a_target + 1e-12
            denom1 = a_ans_t + a_target_t + 1e-12

            # p = attention fraction on target vs {target, answer}
            p0 = a_target / denom0
            p1 = a_target_t / denom1

            all_delta_p.append(p1 - p0)  # in [-1, 1]

            # "flip": argmax moves from answer → target
            arg0 = int(alpha.argmax())
            arg1 = int(alpha_tilde.argmax())
            all_flip.append(int(arg0 == ans_tok and arg1 == target_tok))

            all_orig_a.append(a_ans)
            all_orig_a_target.append(a_target)
            all_interv_a.append(a_ans_t)
            all_interv_a_target.append(a_target_t)

        all_attn_orig.append(attn_base_np)
        all_attn_interv.append(attn_int_np)

        num_batches_run += 1

    all_delta_p = np.array(all_delta_p)
    all_flip = np.array(all_flip)
    all_orig_a = np.array(all_orig_a)
    all_orig_a_target = np.array(all_orig_a_target)
    all_interv_a = np.array(all_interv_a)
    all_interv_a_target = np.array(all_interv_a_target)

    all_attn_orig = np.concatenate(all_attn_orig, axis=0)
    all_attn_interv = np.concatenate(all_attn_interv, axis=0)

    return {
        "mean_delta_p": float(all_delta_p.mean()),
        "std_delta_p": float(all_delta_p.std()),
        "flip_rate": float(all_flip.mean()),
        "num_examples": int(len(all_delta_p)),
        "mean_orig_a": float(all_orig_a.mean()),
        "mean_orig_a_target": float(all_orig_a_target.mean()),
        "mean_interv_a": float(all_interv_a.mean()),
        "mean_interv_a_target": float(all_interv_a_target.mean()),
        "mean_attn_orig": all_attn_orig.mean(axis=0),
        "mean_attn_interv": all_attn_interv.mean(axis=0),
    }


# %%


seed = 11
seed_all(seed)
num_instances = 9
max_objects = 9
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
order_dataset = BindingDataset.from_sampler(order_sampler, num_samples)
lexical_dataset = BindingDataset.from_sampler(lexical_sampler, num_samples)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
convert_to_hooked_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# %%

# Llama
filtered_heads = [
    (16, 1),
    (17, 24),
    (20, 1),

    (16, 8),
    (16, 19),
    (20, 13),
    (20, 14),
    (22, 14),
    (24, 27),
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

# %%


all_metrics = {}
num_test_samples = 1000

interv_data_sampler = partial(
    build_prompt,
    schema=schema,
    num_instances=num_instances,
    query_cat_idx=query_cat_id,
    answer_cat_idx=answer_cat_id,
    query_from_unused=True,
)
interv_dataset = BindingDataset.from_sampler(interv_data_sampler, num_test_samples)[:num_test_samples]["input"]
for head in tqdm(filtered_heads):
    print(f"=== Head L{head[0]}H{head[1]} ===")
    hook_layer = head[0]
    head_idx = head[1]

    U_order, S_order, Vh_order, rank_order = util_funcs.get_qk_subspace(
        C_order_pos[head], C_order_neg[head], thresh=0.99
    )
    U_lex, S_lex, Vh_lex, rank_lex = util_funcs.get_qk_subspace(
        C_lex_pos[head], C_lex_neg[head], thresh=0.99
    )

    all_metrics[(head, "order")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_order,
        Vh=Vh_order,
        rank=rank_order,
        num_instances=num_instances,
        batch_size=batch_size,
    )
    all_metrics[(head, "lexical")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_lex,
        Vh=Vh_lex,
        rank=rank_lex,
        num_instances=num_instances,
        batch_size=batch_size,
    )
    all_metrics[(head, "both")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_order,
        Vh=Vh_order,
        rank=rank_order,
        num_instances=num_instances,
        batch_size=batch_size,
        second=(U_lex, Vh_lex, rank_lex),
    )
    all_metrics[(head, "order")]["rank"] = rank_order
    all_metrics[(head, "lexical")]["rank"] = rank_lex
    all_metrics[(head, "both")]["rank"] = rank_order + rank_lex

    basis_rng = np.random.default_rng(seed + hook_layer * 1000 + head_idx)
    order_key_basis = Vh_order.T[:, :rank_order]
    lexical_key_basis = Vh_lex.T[:, :rank_lex]
    order_random_basis = build_random_key_basis(order_key_basis, basis_rng)
    lexical_random_basis = build_random_key_basis(lexical_key_basis, basis_rng)
    assert order_random_basis.shape[1] == rank_order
    assert lexical_random_basis.shape[1] == rank_lex


    all_metrics[(head, "order_baseline")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_order,
        Vh=order_random_basis.T,
        rank=rank_order,
        num_instances=num_instances,
        batch_size=batch_size,
    )
    all_metrics[(head, "lexical_baseline")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_lex,
        Vh=lexical_random_basis.T,
        rank=rank_lex,
        num_instances=num_instances,
        batch_size=batch_size,
    )
    all_metrics[(head, "both_baseline")] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        schema,
        hook_layer=hook_layer,
        head_idx=head_idx,
        U=U_order,
        Vh=order_random_basis.T,
        rank=rank_order,
        num_instances=num_instances,
        batch_size=batch_size,
        second=(U_lex, lexical_random_basis.T, rank_lex),
    )
    all_metrics[(head, "order_baseline")]["rank"] = rank_order
    all_metrics[(head, "lexical_baseline")]["rank"] = rank_lex
    all_metrics[(head, "both_baseline")]["rank"] = rank_order + rank_lex

    for subspace_type in [
        "order",
        "lexical",
        "both",
        "order_baseline",
        "lexical_baseline",
        "both_baseline",
    ]:
        print(f"    === Causal intervention metrics ({subspace_type})===")
        for k, v in all_metrics[(head, subspace_type)].items():
            if k in ["mean_attn_orig", "mean_attn_interv"]:
                continue
            print(f"{k}: {v}")


# %%

import pickle

with open("causal_intervention_results_9_entities.pkl", "wb") as f:
    pickle.dump(all_metrics, f)

# %%

