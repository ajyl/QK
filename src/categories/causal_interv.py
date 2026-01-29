import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hook_utils import convert_to_hooked_model, seed_all
from dgp.entities import LIQUIDS, COUNTRIES, VEHICLES, ANIMALS, FRUITS
from categories.C_filter_heads import build_C_filter_heads, index_query_key_vecs
from categories.filter_head_utils import build_filter_dataset
from categories.filter_head_utils import collect_item_timesteps_from_input_ids
from categories.filter_head_utils import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_SELECT_ALL,
    _make_bullets,
    make_plural,
)
from util_funcs import (
    format_prompts,
    get_qk_subspace,
    run_forward_pass,
    remove_all_hooks,
)
from constants import (
    BASE_DIR,
    key_module_name,
    query_module_name,
    attn_module_name,
    qk_module_name,
)

# %%

CATEGORIES = {
    "animal": ANIMALS,
    "fruit": FRUITS,
    "liquid": LIQUIDS,
    "vehicle": VEHICLES,
    "country": COUNTRIES,
}


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


def edit_qk(q_vecs, k_orig, k_target, V_r):
    coeff_orig = k_orig @ V_r
    coeff_target = k_target @ V_r

    orig_edited = k_orig + (coeff_target - coeff_orig) @ V_r.T
    target_edited = k_target + (coeff_orig - coeff_target) @ V_r.T

    new_logits_orig = (q_vecs * orig_edited).sum(dim=-1)
    new_logits_target = (q_vecs * target_edited).sum(dim=-1)
    return new_logits_orig, new_logits_target


def edit_qk_with_method(
    q_vecs,
    k_orig,
    k_target,
    method,
    V_r_global=None,
    V_r_orig=None,
    V_r_target=None,
):
    if method == "global":
        if V_r_global is None:
            raise ValueError("edit_qk_with_method(global) requires V_r_global.")
        return edit_qk(q_vecs, k_orig, k_target, V_r_global)

    if method == "orig_only":
        if V_r_orig is None:
            raise ValueError("edit_qk_with_method(orig_only) requires V_r_orig.")
        return edit_qk(q_vecs, k_orig, k_target, V_r_orig)

    if method == "swap_both":
        if V_r_orig is None or V_r_target is None:
            raise ValueError(
                "edit_qk_with_method(swap_both) requires V_r_orig/V_r_target."
            )
        coeff_orig_in_orig = k_orig @ V_r_orig
        coeff_target_in_orig = k_target @ V_r_orig
        coeff_orig_in_target = k_orig @ V_r_target
        coeff_target_in_target = k_target @ V_r_target

        orig_edited = (
            k_orig + (coeff_target_in_target - coeff_orig_in_target) @ V_r_target.T
        )
        target_edited = (
            k_target + (coeff_orig_in_orig - coeff_target_in_orig) @ V_r_orig.T
        )
        new_logits_orig = (q_vecs * orig_edited).sum(dim=-1)
        new_logits_target = (q_vecs * target_edited).sum(dim=-1)
        return new_logits_orig, new_logits_target

    raise ValueError(f"Unknown edit method: {method}")


def build_item_category_map(categories):
    item_to_category = {}
    for category, items in categories.items():
        for item in items:
            item_to_category[item] = category
    return item_to_category


def limit_predicate_items_in_dataset(dataset, categories, rng):
    new_dataset = []
    for sample in dataset:
        predicate = sample["predicate"]
        pred_items = set(categories[predicate])
        items = sample["items"]
        predicate_items = [w for w in items if w in pred_items]
        if len(predicate_items) <= 1:
            new_dataset.append(sample)
            continue

        keep_item = rng.choice(predicate_items)
        new_items = []
        kept = False
        for w in items:
            if w in pred_items:
                if not kept and w == keep_item:
                    new_items.append(w)
                    kept = True
                continue
            new_items.append(w)

        pos_item_idxs = [i for i, w in enumerate(new_items) if w in pred_items]
        neg_item_idxs = [i for i, w in enumerate(new_items) if w not in pred_items]
        task = sample["task"]

        if task == "select_first":
            ans_idx = min(pos_item_idxs)
            which = "first"
            answer = new_items[ans_idx]
            prompt = PROMPT_TEMPLATE.format(
                bullets=_make_bullets(new_items),
                which=which,
                predicate=predicate,
            )
        elif task == "select_last":
            ans_idx = max(pos_item_idxs)
            which = "last"
            answer = new_items[ans_idx]
            prompt = PROMPT_TEMPLATE.format(
                bullets=_make_bullets(new_items),
                which=which,
                predicate=predicate,
            )
        else:
            answer = [new_items[i] for i in pos_item_idxs]
            prompt = PROMPT_TEMPLATE_SELECT_ALL.format(
                bullets=_make_bullets(new_items),
                predicate=make_plural(predicate),
            )

        new_sample = dict(sample)
        new_sample.update(
            dict(
                raw_input=prompt,
                items=new_items,
                pos_item_idxs=pos_item_idxs,
                neg_item_idxs=neg_item_idxs,
                answer=answer,
            )
        )
        new_dataset.append(new_sample)
    return new_dataset


@torch.no_grad()
def run_causal_interventions_for_head(
    model,
    tokenizer,
    dataset,
    hook_layer,
    head_idx,
    key_basis,
    key_bases_by_category,
    num_key_value_groups,
    batch_size,
    rng,
    item_to_category,
    edit_method="global",
):
    """
    For each example:
      1) Choose a positive (predicate-matching) item and a negative item.
      2) Baseline run: record attention and Q,K.
      3) Offline: edit the target key in the category subspace, compute new qk logits.
      4) Second run: hook into qk logits to inject edited logits.
      5) Measure change in attention mass from positive -> negative.
    """
    remove_all_hooks(model)

    key_name = key_module_name.format(hook_layer)
    query_name = query_module_name.format(hook_layer)
    attn_name = attn_module_name.format(hook_layer)
    qk_name = qk_module_name.format(hook_layer)

    V_r_global = key_basis.to(model.device)

    all_delta_p = []
    all_flip = []
    all_orig_a = []
    all_orig_a_target = []
    all_interv_a = []
    all_interv_a_target = []
    all_attn_orig = []
    all_attn_interv = []

    named_modules = dict(model.named_modules())
    if qk_name not in named_modules:
        raise ValueError(
            f"{qk_name} not found in model.named_modules(). "
            "You need a HookPoint there (e.g. in convert_to_hooked_model)."
        )

    for batch_idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_idx : batch_idx + batch_size]
        if len(batch) == 0:
            continue

        prompts = format_prompts(tokenizer, [sample["raw_input"] for sample in batch])
        record_names = [key_name, query_name, attn_name]
        cache, _, input_ids, attention_mask = run_forward_pass(
            model, tokenizer, record_names, prompts
        )

        item_pos = collect_item_timesteps_from_input_ids(
            tokenizer, input_ids, [x["items"] for x in batch], prefer_last_token=True
        )
        item_pos = torch.tensor(item_pos, device=model.device)

        attn_base = cache[attn_name][0][:, head_idx, -1, :]  # [B, S]

        orig_item_idxs = []
        target_item_idxs = []
        for b, sample in enumerate(batch):
            pos_choices = torch.tensor(
                sample["pos_item_idxs"], device=model.device, dtype=torch.long
            )
            neg_choices = sample["neg_item_idxs"]

            pos_token_idxs = item_pos[b, pos_choices]
            pos_attn = attn_base[b, pos_token_idxs]
            max_pos_idx = int(torch.argmax(pos_attn).item())
            orig_item_idxs.append(int(pos_choices[max_pos_idx].item()))
            target_item_idxs.append(rng.choice(neg_choices))

        orig_item_idxs = torch.tensor(orig_item_idxs, device=model.device)
        target_item_idxs = torch.tensor(target_item_idxs, device=model.device)

        orig_token_idxs = item_pos.gather(dim=1, index=orig_item_idxs[:, None]).squeeze(
            1
        )
        target_token_idxs = item_pos.gather(
            dim=1, index=target_item_idxs[:, None]
        ).squeeze(1)

        q_vecs, k_vecs = index_query_key_vecs(
            cache,
            hook_layer,
            item_pos,
            num_key_value_groups,
        )
        q_vecs = q_vecs[:, head_idx, :].to(model.device)  # [B, D]
        k_vecs = k_vecs[:, head_idx, :, :].to(model.device)  # [B, M, D]

        b_idx = torch.arange(len(batch), device=model.device)
        k_orig = k_vecs[b_idx, orig_item_idxs, :]  # [B, D]
        k_target = k_vecs[b_idx, target_item_idxs, :]  # [B, D]

        l_orig_new_vals = []
        l_target_new_vals = []
        for b in range(len(batch)):
            V_r_orig = None
            V_r_target = None
            if edit_method in {"orig_only", "swap_both"}:
                orig_category = batch[b]["predicate"]
                V_r_orig = key_bases_by_category[orig_category].to(model.device)
            if edit_method == "swap_both":
                target_item = batch[b]["items"][int(target_item_idxs[b].item())]
                target_category = item_to_category[target_item]
                V_r_target = key_bases_by_category[target_category].to(model.device)

            l_orig_new_b, l_target_new_b = edit_qk_with_method(
                q_vecs[b : b + 1],
                k_orig[b : b + 1],
                k_target[b : b + 1],
                edit_method,
                V_r_global=V_r_global,
                V_r_orig=V_r_orig,
                V_r_target=V_r_target,
            )
            l_orig_new_vals.append(l_orig_new_b.squeeze(0))
            l_target_new_vals.append(l_target_new_b.squeeze(0))
        l_orig_new = torch.stack(l_orig_new_vals, dim=0)
        l_target_new = torch.stack(l_target_new_vals, dim=0)

        attn_int_cache = []

        def qk_hook(_module, _inputs, output):
            qk = output
            B_local, H_local, Q_local, _K_local = qk.shape
            assert B_local == len(batch)
            assert H_local > head_idx
            q_pos = Q_local - 1  # final position is the query
            b_local = torch.arange(B_local, device=qk.device)
            qk[b_local, head_idx, q_pos, orig_token_idxs] = l_orig_new.to(qk.device)
            qk[b_local, head_idx, q_pos, target_token_idxs] = l_target_new.to(qk.device)
            return qk

        def attn_hook(_module, _inputs, output):
            attn_int_cache.append(output.detach().cpu())
            return output

        qk_module = named_modules[qk_name]
        attn_module = named_modules[attn_name]

        h1 = qk_module.register_forward_hook(qk_hook)
        h2 = attn_module.register_forward_hook(attn_hook)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        h1.remove()
        h2.remove()

        attn_int = attn_int_cache[0][:, head_idx, -1, :]  # [B, S]

        attn_base_np = attn_base.cpu().numpy()
        attn_int_np = attn_int.cpu().numpy()

        for b in range(len(batch)):
            orig_tok = orig_token_idxs[b].item()
            target_tok = target_token_idxs[b].item()

            alpha = attn_base_np[b]
            alpha_tilde = attn_int_np[b]

            a_orig = alpha[orig_tok]
            a_target = alpha[target_tok]
            a_orig_t = alpha_tilde[orig_tok]
            a_target_t = alpha_tilde[target_tok]

            denom0 = a_orig + a_target + 1e-12
            denom1 = a_orig_t + a_target_t + 1e-12

            p0 = a_target / denom0
            p1 = a_target_t / denom1

            all_delta_p.append(p1 - p0)

            arg0 = int(alpha.argmax())
            arg1 = int(alpha_tilde.argmax())
            all_flip.append(int(arg0 == orig_tok and arg1 == target_tok))

            all_orig_a.append(a_orig)
            all_orig_a_target.append(a_target)
            all_interv_a.append(a_orig_t)
            all_interv_a_target.append(a_target_t)

        all_attn_orig.append(attn_base_np)
        all_attn_interv.append(attn_int_np)

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

model_name = "meta-llama/Llama-3.1-8B-Instruct"
seed = 11
batch_size = 16
task = "select_all"

subspace_num_samples = 2000
n_per_category = 5
subspace_thresh = 0.99

interv_num_samples = 1000
heads = [(16, 19), (20, 26), (20, 14)]


output_dir = Path(BASE_DIR) / "figures" / "category_causal_intervention"
output_dir.mkdir(parents=True, exist_ok=True)

# %%

seed_all(seed)
rng = np.random.default_rng(seed)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
convert_to_hooked_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

hook_layers = sorted(list({layer for (layer, _) in heads}))
record_module_names = [
    key_module_name.format(hook_layer) for hook_layer in hook_layers
] + [query_module_name.format(hook_layer) for hook_layer in hook_layers]

categories = sorted(CATEGORIES.keys())

# %%

subspace_samples = build_filter_dataset(
    categories=CATEGORIES,
    n_samples=subspace_num_samples,
    n_per_category=n_per_category,
    task=task,
    seed=seed,
)
C_pos, C_neg, _, counts, _ = build_C_filter_heads(
    model,
    tokenizer,
    record_module_names,
    subspace_samples,
    heads,
    batch_size,
    categories=categories,
)

num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups

head_bases = {}
item_to_category = build_item_category_map(CATEGORIES)
for head in heads:
    if int(counts[head].sum().item()) == 0:
        continue

    _keys = []
    _queries = []
    key_bases_by_category = {}
    for c_idx in range(len(categories)):
        U, _S, Vh, rank = get_qk_subspace(
            C_pos[head][c_idx],
            C_neg[head][c_idx],
            thresh=subspace_thresh,
        )
        _keys.append(Vh[:rank, :])
        _queries.append(U[:, :rank])
        key_bases_by_category[categories[c_idx]] = Vh[:rank, :].T

    key_basis = pos_qr(torch.cat(_keys, dim=0).T)
    query_basis = pos_qr(torch.cat(_queries, dim=1))
    rank = key_basis.shape[1]
    assert rank == query_basis.shape[1]
    head_bases[head] = (query_basis, key_basis, rank, key_bases_by_category)

# %%

interv_dataset = build_filter_dataset(
    categories=CATEGORIES,
    n_samples=1000,
    n_per_category=n_per_category,
    task=task,
    seed=seed + 1,
)
interv_dataset = limit_predicate_items_in_dataset(interv_dataset, CATEGORIES, rng)
edit_method = "global"

all_metrics = {}
for head in tqdm(heads):
    if head not in head_bases:
        print(f"Head {head}: no category subspace found, skipping.")
        continue

    hook_layer, head_idx = head
    query_basis, key_basis, rank, key_bases_by_category = head_bases[head]

    rng_state = rng.bit_generator.state
    all_metrics[head] = run_causal_interventions_for_head(
        model,
        tokenizer,
        interv_dataset,
        hook_layer=hook_layer,
        head_idx=head_idx,
        key_basis=key_basis[:, :rank],
        key_bases_by_category=key_bases_by_category,
        num_key_value_groups=num_key_value_groups,
        batch_size=batch_size,
        rng=rng,
        item_to_category=item_to_category,
        edit_method=edit_method,
    )
    rng_state_after = rng.bit_generator.state
    all_metrics[head]["rank"] = rank
    all_metrics[head]["edit_method"] = edit_method

    if edit_method == "global":
        rng.bit_generator.state = rng_state
        basis_rng = np.random.default_rng(seed + hook_layer * 1000 + head_idx)
        random_key_basis = build_random_key_basis(key_basis[:, :rank], basis_rng)
        random_metrics = run_causal_interventions_for_head(
            model,
            tokenizer,
            interv_dataset,
            hook_layer=hook_layer,
            head_idx=head_idx,
            key_basis=random_key_basis,
            key_bases_by_category=key_bases_by_category,
            num_key_value_groups=num_key_value_groups,
            batch_size=batch_size,
            rng=rng,
            item_to_category=item_to_category,
            edit_method=edit_method,
        )
        rng.bit_generator.state = rng_state_after
        random_metrics["rank"] = rank
        random_metrics["edit_method"] = edit_method
        all_metrics[head]["random_baseline"] = random_metrics

    print(f"=== Head L{hook_layer}H{head_idx} ===")
    for k, v in all_metrics[head].items():
        if k in ["mean_attn_orig", "mean_attn_interv"]:
            continue
        if k == "random_baseline":
            print("random_baseline:")
            for rbk in [
                "mean_orig_a",
                "mean_orig_a_target",
                "mean_interv_a",
                "mean_interv_a_target",
                "mean_delta_p",
                "flip_rate",
            ]:
                print(f"  {rbk}: {v[rbk]}")
            continue
        print(f"{k}: {v}")

# %%

import pickle

out_path = output_dir / "category_causal_intervention_results.pkl"
with open(out_path, "wb") as f:
    pickle.dump(all_metrics, f)

print(f"Saved results to: {out_path}")

# %%
