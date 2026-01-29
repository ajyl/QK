import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv
from dgp.build_data import build_counterfactual_lexical
from util_funcs import run_forward_pass, format_prompts, to_str_tokens
from constants import key_module_name, query_module_name


def index_query_key_vecs_lex(
    cache, layer_idx, head_idx, timesteps, num_key_value_groups
):

    q_vecs = cache[query_module_name.format(layer_idx)][0]
    q_vecs = q_vecs[:, head_idx, -1, :]  # [B,d]
    k_vecs = cache[key_module_name.format(layer_idx)][0]
    k_vecs = repeat_kv(k_vecs, num_key_value_groups)  # [B,H,seq,d]
    k_vecs = k_vecs[:, head_idx, :, :]  # [B,seq,d]

    batch_size = k_vecs.shape[0]
    b_idx = torch.arange(batch_size, device=k_vecs.device)

    k_vecs = k_vecs[b_idx, timesteps, :]  # [B,d]

    return q_vecs, k_vecs


def _collect_lex_key_timesteps(prompts, tokenizer, key_matcher):
    key_timesteps = []

    for sample_idx, prompt in enumerate(prompts):

        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)

        _key_timesteps = []
        _cf_key_timesteps = []

        for token_idx, token in enumerate(prompt_str_tokenized):
            if key_matcher(token):
                _key_timesteps.append(token_idx)

        key_timesteps.append(_key_timesteps)

    assert len(key_timesteps) == len(prompts)
    return key_timesteps


def _get_lex_ans_timesteps(prompts, query_idxs, tokenizer, key_matcher):
    key_timesteps = _collect_lex_key_timesteps(prompts, tokenizer, key_matcher)
    # [B]
    ans_timesteps = torch.tensor(
        [key_timesteps[b_idx][query_idxs[b_idx]] for b_idx in range(len(prompts))],
    )
    return ans_timesteps


def get_C_lex_batch(
    model,
    tokenizer,
    cache,
    prompts,
    query_idxs,
    key_matcher,
    filtered_heads,
):

    d_head = model.config.head_dim
    ans_timesteps = _get_lex_ans_timesteps(
        prompts, query_idxs, tokenizer, key_matcher
    ).to(model.device)

    num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups

    C, query_vecs, key_vecs, attn_patterns = {}, {}, {}, {}
    for curr_head in filtered_heads:
        hook_layer, head_idx = curr_head
        q_vecs, k_vecs = index_query_key_vecs_lex(
            cache,
            hook_layer,
            head_idx,
            ans_timesteps,
            num_key_value_groups,
        )

        query_vecs[curr_head] = q_vecs.cpu()
        key_vecs[curr_head] = k_vecs.cpu()
        C[curr_head] = torch.einsum("bq,bk->qk", q_vecs, k_vecs).cpu()
        # attn_patterns[curr_head] = cache[attn_module_name.format(hook_layer)][0][
        #    :, head_idx, -1
        # ].cpu()
    return C, query_vecs, key_vecs, attn_patterns


def _build_C_lexical(
    model,
    tokenizer,
    record_module_names,
    dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
):

    d_head = model.config.head_dim

    key_matcher = schema.matchers[answer_cat_id]

    C_pos = {_head: torch.zeros(d_head, d_head) for _head in filtered_heads}
    C_neg = {_head: torch.zeros(d_head, d_head) for _head in filtered_heads}

    q_vecs_pos = {_head: [] for _head in filtered_heads}
    q_vecs_neg = {_head: [] for _head in filtered_heads}
    k_vecs_pos = {_head: [] for _head in filtered_heads}
    k_vecs_neg = {_head: [] for _head in filtered_heads}

    labels = []
    object_vocab = schema.items["Object"]

    for batch_idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_idx : batch_idx + batch_size]["input"]
        counterfactuals = [
            build_counterfactual_lexical(sample, schema) for sample in batch
        ]

        B = len(batch)
        if B == 0:
            continue

        prompts = format_prompts(tokenizer, [sample["raw_input"] for sample in batch])
        cf_prompts = format_prompts(
            tokenizer, [sample["raw_input"] for sample in counterfactuals]
        )
        pos_cache, _, _, _ = run_forward_pass(model, tokenizer, record_module_names, prompts)
        neg_cache, _, _, _ = run_forward_pass(model, tokenizer, record_module_names, cf_prompts)

        query_idxs = [sample["queryIndex"] for sample in batch]
        _C_pos, _q_vecs_pos, _k_vecs_pos, _attn_patterns = get_C_lex_batch(
            model,
            tokenizer,
            pos_cache,
            prompts,
            query_idxs,
            key_matcher,
            filtered_heads,
        )
        _C_neg, _q_vecs_neg, _k_vecs_neg, _ = get_C_lex_batch(
            model,
            tokenizer,
            neg_cache,
            cf_prompts,
            query_idxs,
            key_matcher,
            filtered_heads,
        )
        labels_str = [
            batch[b_idx][f"Object.{query_cat_id}.{x}"]
            for b_idx, x in enumerate(query_idxs)
        ]
        _labels = [object_vocab.index(label_str) for label_str in labels_str]
        labels.extend(_labels)

        for _head in filtered_heads:
            C_pos[_head] += _C_pos[_head]
            C_neg[_head] += _C_neg[_head]
            q_vecs_pos[_head].append(_q_vecs_pos[_head])
            k_vecs_pos[_head].append(_k_vecs_pos[_head])
            q_vecs_neg[_head].append(_q_vecs_neg[_head])
            k_vecs_neg[_head].append(_k_vecs_neg[_head])
            # attn_patterns[_head].append(_attn_patterns[_head])

    labels = torch.tensor(labels, dtype=torch.long)
    return C_pos, C_neg, q_vecs_pos, q_vecs_neg, k_vecs_pos, k_vecs_neg, labels


def build_C_lexical(
    model,
    tokenizer,
    record_module_names,
    dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
):

    (
        C_lex_pos,
        C_lex_neg,
        query_vecs_lex_pos,
        query_vecs_lex_neg,
        key_vecs_lex_pos,
        key_vecs_lex_neg,
        all_lexical_labels,
    ) = _build_C_lexical(
        model,
        tokenizer,
        record_module_names,
        dataset,
        schema,
        answer_cat_id,
        query_cat_id,
        filtered_heads,
        batch_size,
    )

    total_N = len(dataset)
    for _head in filtered_heads:
        C_lex_pos[_head] /= total_N
        C_lex_neg[_head] /= total_N

        # Lex
        query_vecs_lex_pos[_head] = torch.cat(query_vecs_lex_pos[_head], dim=0)
        key_vecs_lex_pos[_head] = torch.cat(key_vecs_lex_pos[_head], dim=0)
        query_vecs_lex_neg[_head] = torch.cat(query_vecs_lex_neg[_head], dim=0)
        key_vecs_lex_neg[_head] = torch.cat(key_vecs_lex_neg[_head], dim=0)

    return (
        C_lex_pos,
        C_lex_neg,
        query_vecs_lex_pos,
        query_vecs_lex_neg,
        key_vecs_lex_pos,
        key_vecs_lex_neg,
        all_lexical_labels,
    )
