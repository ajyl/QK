import random
import torch
from hook_utils import record_activations

QUERY_CAT_ID = 0
ANSWER_CAT_ID = 1


def format_prompts(tokenizer, prompts):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    formatted = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msg in messages
    ]
    return formatted


def to_str_tokens(tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    return [
        x.replace("▁", " ").replace("Ġ", " ")
        for x in tokenizer.convert_ids_to_tokens(tokens)
    ]


def parse_binding_batch(batch, schema, tokenizer):
    """
    Given a batch of samples, parse and extract the token indices for key and query entities.

    Args:
        batch (list): A batch of samples, each containing 'raw_input' and 'numInstances'.
        schema (Schema): The schema object containing matchers for categories.
        tokenizer (Tokenizer): The tokenizer used to tokenize the prompts.

    Returns:
        tuple: Three lists containing prompts, token indices for key entities, and query entities respectively.
    """
    key_matcher = schema.matchers[ANSWER_CAT_ID]
    query_matcher = schema.matchers[QUERY_CAT_ID]

    prompts = format_prompts(tokenizer, [sample["raw_input"] for sample in batch])

    key_entity_token_idxs = []
    query_entity_token_idxs = []

    for sample_idx, prompt in enumerate(prompts):
        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        sample = batch[sample_idx]
        _key_entity_pos_idxs = []
        _query_entity_pos_idxs = []
        for token_idx, token in enumerate(prompt_str_tokenized):
            if key_matcher(token):
                _key_entity_pos_idxs.append(token_idx)
            if query_matcher(token):
                _query_entity_pos_idxs.append(token_idx)

        assert len(_key_entity_pos_idxs) == sample["numInstances"]
        assert len(_query_entity_pos_idxs) == sample["numInstances"] + 1
        _query_entity_pos_idxs = _query_entity_pos_idxs[:-1]

        key_entity_token_idxs.append(_key_entity_pos_idxs)
        query_entity_token_idxs.append(_query_entity_pos_idxs)

    return prompts, key_entity_token_idxs, query_entity_token_idxs


@torch.no_grad()
def run_forward_pass(model, tokenizer, record_module_names, prompts):
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    with record_activations(model, record_module_names) as cache:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    return cache, logits, input_ids, attention_mask


def get_qk_subspace(C_pos, C_neg, thresh=0.99):
    C_delta = C_pos - C_neg
    U, S, Vh = torch.linalg.svd(C_delta)
    S_sqr = S**2
    expl_var = S_sqr / S_sqr.sum(dim=-1, keepdim=True)
    cum_var = expl_var.cumsum(dim=-1)
    rank = (cum_var < thresh).sum(dim=-1) + 1
    return U, S, Vh, rank.item()


def remove_all_hooks(module: torch.nn.Module):
    # Iterate over all modules, including the current one and its children
    for m in module.modules():
        # Clear forward hooks
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        # Clear forward pre-hooks
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        # Clear backward hooks
        if hasattr(m, "_backward_hooks"):
            m._backward_hooks.clear()
        # Clear full backward hooks (if applicable)
        if hasattr(m, "_full_backward_hooks"):
            m._full_backward_hooks.clear()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def effective_rank_from_svals(s: torch.Tensor, energy: float = 0.99) -> int:
    # s: singular values (descending)
    # energy: fraction of Frobenius energy (sum s^2)
    s2 = s**2
    total = s2.sum().clamp_min(1e-12)
    cum = torch.cumsum(s2, dim=0)
    r = int(torch.searchsorted(cum, energy * total).item()) + 1
    return min(r, s.numel())
