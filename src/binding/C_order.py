import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv
from util_funcs import run_forward_pass, format_prompts, to_str_tokens
from constants import key_module_name, query_module_name


def index_query_key_vecs_order(
    cache,
    layer_idx,
    pos_timesteps,
    neg_timesteps,
    num_key_value_groups,
    num_instances,
):
    # [batch, n_heads, seq, d_head]
    q_vecs = cache[query_module_name.format(layer_idx)][0]
    q_vecs = q_vecs[:, :, -1, :]

    # [batch, n_heads / num_key_value_groups, seq, d_head]
    k_vecs = cache[key_module_name.format(layer_idx)][0]
    # [batch, n_heads, seq, d_head]
    k_vecs = repeat_kv(k_vecs, num_key_value_groups)

    B, H, S, D = k_vecs.shape
    M = num_instances - 1

    pos_idx = pos_timesteps[:, None].expand(B, H)
    b_idx = torch.arange(B, device=k_vecs.device)[:, None]
    h_idx = torch.arange(H, device=k_vecs.device)[None, :]
    pos_k = k_vecs[b_idx, h_idx, pos_idx, :]

    neg_idx = neg_timesteps[:, None, :].expand(B, H, M)
    neg_k = k_vecs.gather(
        2,
        neg_idx.unsqueeze(-1).expand(B, H, M, D),
    )
    return q_vecs, k_vecs, pos_k, neg_k


def collect_key_query_timesteps(prompts, tokenizer, key_matcher, query_matcher):
    key_timesteps = []
    query_timesteps = []

    for prompt in prompts:
        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        _key_timesteps = []
        _query_timesteps = []
        for token_idx, token in enumerate(prompt_str_tokenized):
            if key_matcher(token):
                _key_timesteps.append(token_idx)
            if query_matcher(token):
                _query_timesteps.append(token_idx)

        key_timesteps.append(_key_timesteps)
        query_timesteps.append(_query_timesteps)

    return key_timesteps, query_timesteps


def get_C_order_batch(
    model,
    tokenizer,
    cache,
    prompts,
    query_idxs,
    key_matcher,
    query_matcher,
    filtered_heads,
    num_instances,
):

    key_timesteps, query_timesteps = collect_key_query_timesteps(
        prompts, tokenizer, key_matcher, query_matcher
    )
    assert len(key_timesteps) == len(prompts)
    assert len(query_timesteps) == len(prompts)
    assert len(key_timesteps[0]) == num_instances
    assert len(query_timesteps[0]) == num_instances + 1

    ans_timesteps = torch.tensor(
        [key_timesteps[i][query_idxs[i]] for i in range(len(key_timesteps))],
    ).to(model.device)
    non_ans_timesteps = torch.tensor(
        [
            [x for x in key_timesteps[i] if x != ans_timesteps[i]]
            for i in range(len(key_timesteps))
        ],
        device=model.device,
    )

    probe_key_vecs = {_head: [] for _head in filtered_heads}
    probe_query_vecs = {_head: [] for _head in filtered_heads}
    C_pos = {}
    C_neg = {}
    order_id_labels = []
    query_labels = []
    for curr_head in filtered_heads:
        hook_layer, head_idx = curr_head
        q_vecs, k_vecs, pos_k, neg_k = index_query_key_vecs_order(
            cache,
            hook_layer,
            ans_timesteps,
            non_ans_timesteps,
            model.model.layers[0].self_attn.num_key_value_groups,
            num_instances,
        )
        for b, box_pos_list in enumerate(key_timesteps):
            for j, pos in enumerate(box_pos_list):
                probe_key_vecs[curr_head].append(
                    k_vecs[b, head_idx, pos].detach().cpu()
                )
                probe_query_vecs[curr_head].append(q_vecs[b, head_idx].detach().cpu())

        C_pos_batch = torch.einsum("bq,bk->qk", q_vecs[:, head_idx], pos_k[:, head_idx])
        C_neg_batch = torch.einsum(
            "bq,bmk->mqk", q_vecs[:, head_idx], neg_k[:, head_idx]
        ).mean(dim=0)
        C_pos[curr_head] = C_pos_batch.cpu()
        C_neg[curr_head] = C_neg_batch.cpu()
    for b, box_pos_list in enumerate(key_timesteps):
        order_id_labels.extend(list(range(len(box_pos_list))))
        query_labels.extend([query_idxs[b]] * len(box_pos_list))
    return C_pos, C_neg, probe_key_vecs, probe_query_vecs, order_id_labels, query_labels


def _build_C_order(
    model,
    tokenizer,
    record_module_names,
    dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
    num_instances,
):

    d_head = model.config.head_dim
    key_matcher = schema.matchers[answer_cat_id]
    query_matcher = schema.matchers[query_cat_id]

    C_pos = {_head: torch.zeros(d_head, d_head) for _head in filtered_heads}
    C_neg = {_head: torch.zeros(d_head, d_head) for _head in filtered_heads}
    k_vecs = {_head: [] for _head in filtered_heads}
    q_vecs = {_head: [] for _head in filtered_heads}
    labels = []
    q_labels = []

    for batch_idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_idx : batch_idx + batch_size]["input"]

        B = len(batch)
        if B == 0:
            continue

        prompts = format_prompts(tokenizer, [sample["raw_input"] for sample in batch])
        cache, _, _, _ = run_forward_pass(model, tokenizer, record_module_names, prompts)

        query_idxs = [sample["queryIndex"] for sample in batch]

        _C_pos, _C_neg, _k_vecs, _q_vecs, _order_id_labels, _q_labels = (
            get_C_order_batch(
                model,
                tokenizer,
                cache,
                prompts,
                query_idxs,
                key_matcher,
                query_matcher,
                filtered_heads,
                num_instances,
            )
        )
        labels.extend(_order_id_labels)
        q_labels.extend(_q_labels)

        for _head in filtered_heads:
            C_pos[_head] += _C_pos[_head]
            C_neg[_head] += _C_neg[_head]
            q_vecs[_head].extend(_q_vecs[_head])
            k_vecs[_head].extend(_k_vecs[_head])

    return C_pos, C_neg, q_vecs, k_vecs, labels, q_labels


def build_C_order(
    model,
    tokenizer,
    record_module_names,
    dataset,
    schema,
    answer_cat_id,
    query_cat_id,
    filtered_heads,
    batch_size,
    num_instances,
):
    (
        C_order_pos,
        C_order_neg,
        query_vecs_order,
        key_vecs_order,
        all_order_labels,
        q_labels,
    ) = _build_C_order(
        model,
        tokenizer,
        record_module_names,
        dataset,
        schema,
        answer_cat_id,
        query_cat_id,
        filtered_heads,
        batch_size,
        num_instances,
    )

    total_N = len(dataset)
    for _head in filtered_heads:
        C_order_pos[_head] /= total_N
        C_order_neg[_head] /= total_N

        # Order
        query_vecs_order[_head] = torch.stack(query_vecs_order[_head], dim=0)
        key_vecs_order[_head] = torch.stack(key_vecs_order[_head], dim=0)

    return (
        C_order_pos,
        C_order_neg,
        query_vecs_order,
        key_vecs_order,
        all_order_labels,
        q_labels,
    )
