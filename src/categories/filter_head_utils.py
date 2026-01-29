import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv

from hook_utils import convert_to_hooked_model
from util_funcs import run_forward_pass, format_prompts
from dgp.entities import LIQUIDS, COUNTRIES, VEHICLES, ANIMALS, FRUITS, SPORTS, BIRDS, FISH, CITIES, FOOD, STATES
from constants import key_module_name, query_module_name, attn_module_name



CATEGORIES = {
    "animal": ANIMALS,
    "fruit": FRUITS,
    "liquid": LIQUIDS,
    "vehicle": VEHICLES,
    "country": COUNTRIES,
    #
    "sport": SPORTS,
    "bird": BIRDS,
    "fish": FISH,
    "cities": CITIES,
    "food": FOOD,
    "states": STATES,
}

PROMPT_TEMPLATE = """Here is a list:
{bullets}
Respond in one word, only the answer and nothing else: What is the {which} {predicate} in the list? Answer:"""


PROMPT_TEMPLATE_SELECT_ALL = """Here is a list:
{bullets}
Respond in single words, only the answers and nothing else: What are the {predicate} in the list? Answer:"""


def _make_bullets(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items])


# ---------- robust subseq search on *actual* padded input_ids ----------
def _find_subseq(haystack: List[int], needle: List[int]) -> int:
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def collect_item_timesteps_from_input_ids(
    tokenizer,
    input_ids: torch.Tensor,  # [B, S] (padded, exactly what model saw)
    items_by_prompt: List[List[str]],
    prefer_last_token: bool = True,
) -> List[List[int]]:
    """
    Returns padded-seq token indices aligned with `input_ids` (so they work for attn patterns).
    Tries a few common prefixes since list items may appear after '- ' or newlines.
    """
    B, S = input_ids.shape
    input_ids_list = input_ids.tolist()

    # try several textual prefixes that commonly precede items in your bullet list format
    prefixes = ["\n- ", "- ", "\n", " ", ""]

    out_positions: List[List[int]] = []
    for b in range(B):
        row = input_ids_list[b]
        positions = []
        for item in items_by_prompt[b]:
            start = -1
            item_tok = None

            for pref in prefixes:
                tok = tokenizer(pref + item, add_special_tokens=False).input_ids
                s = _find_subseq(row, tok)
                if s != -1:
                    start = s
                    item_tok = tok
                    break

            if start == -1:
                # final fallback: search for the item alone
                tok = tokenizer(item, add_special_tokens=False).input_ids
                start = _find_subseq(row, tok)
                item_tok = tok

            if start == -1:
                raise ValueError(
                    f"Could not locate item '{item}' in tokenized prompt row {b}."
                )

            pos = start + (len(item_tok) - 1 if prefer_last_token else 0)
            positions.append(pos)

        out_positions.append(positions)
    return out_positions


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Works for both left-padding and right-padding.
    attention_mask: [B,S] with 1 for real tokens, 0 for pad.
    Returns: [B] indices of last token where mask==1.
    """
    B, S = attention_mask.shape
    # find first 1 in the reversed mask => distance from end
    dist_from_end = attention_mask.flip(1).long().argmax(dim=1)  # [B]
    return (S - 1) - dist_from_end


def get_last_query_attention_from_cache(
    cache: Dict[str, List[torch.Tensor]],
    layer_idx: int,
) -> torch.Tensor:
    """
    Reads attention patterns from cache.
    Expects cache[attn_module_name.format(layer_idx)][0] is [B, H, Q, K] (often Q=K=S).
    Returns last-query attention: [B, H, K]
    """
    attn = cache[attn_module_name.format(layer_idx)][0]  # [B,H,Q,K]
    B, H, Q, K = attn.shape
    attn_last = attn[:, :, -1]
    return attn_last


def make_plural(category):
    if category.endswith("y"):
        return category[:-1] + "ies"
    else:
        return category + "s"


def build_filter_dataset(
    categories,
    n_samples: int,
    n_per_category: int = 2,  # how many of each class appear in the list
    task: str = "select_all",  # "select_first" or "select_last"
    seed: int = 0,
) -> List[dict]:
    """
    Returns a list of samples. Each sample contains:
      - raw_input: prompt string
      - predicate: one of {"animal","fruit","flower"}
      - items: the shuffled list items
      - pos_item_idxs: indices into `items` that satisfy predicate
      - neg_item_idxs: indices into `items` that do not satisfy predicate
      - answer: correct answer word for the reduce step
    """
    assert task in {"select_first", "select_last", "select_all"}
    rng = random.Random(seed)

    all_items = []
    for cat, words in categories.items():
        # if n_per_category > len(words), you'll want to expand vocab here
        assert n_per_category <= len(words)
        all_items.extend(words[:n_per_category])

    samples = []
    for _ in range(n_samples):
        predicate = rng.choice(list(categories.keys()))
        items = all_items.copy()
        rng.shuffle(items)

        pos_item_idxs = [i for i, w in enumerate(items) if w in categories[predicate]]
        neg_item_idxs = [
            i for i, w in enumerate(items) if w not in categories[predicate]
        ]

        if task == "select_first":
            ans_idx = min(pos_item_idxs)
            which = "first"
        elif task == "select_last":
            ans_idx = max(pos_item_idxs)
            which = "last"
        else:
            ans_idx = pos_item_idxs
            which = None

        if task == "select_all":
            answer = [items[i] for i in pos_item_idxs]
            prompt = PROMPT_TEMPLATE_SELECT_ALL.format(
                bullets=_make_bullets(items),
                # which=which,
                predicate=make_plural(predicate),
            )
        else:
            answer = items[ans_idx]
            prompt = PROMPT_TEMPLATE.format(
                bullets=_make_bullets(items),
                which=which,
                predicate=predicate,
            )

        samples.append(
            dict(
                raw_input=prompt,
                predicate=predicate,
                items=items,
                pos_item_idxs=pos_item_idxs,
                neg_item_idxs=neg_item_idxs,
                answer=answer,
                task=task,
            )
        )
    return samples



def _find_subseq(haystack: List[int], needle: List[int]) -> int:
    """Return start index of `needle` in `haystack`, or -1 if not found."""
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1
    # naive scan; fine for short prompts
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def collect_item_timesteps(
    tokenizer,
    prompts: List[str],
    items_by_prompt: List[List[str]],
    prefer_last_token: bool = True,
) -> List[List[int]]:
    """
    For each prompt, return token indices corresponding to each item occurrence in `items_by_prompt[i]`.
    We search for the tokenization of " " + item (space-prefixed) inside the tokenized prompt.
    """
    out_positions: List[List[int]] = []
    for prompt, items in zip(prompts, items_by_prompt):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

        positions = []
        for item in items:
            item_ids = tokenizer(" " + item, add_special_tokens=False).input_ids
            start = _find_subseq(prompt_ids, item_ids)
            if start == -1:
                # fall back: try without leading space
                item_ids = tokenizer(item, add_special_tokens=False).input_ids
                start = _find_subseq(prompt_ids, item_ids)
            if start == -1:
                raise ValueError(f"Could not locate item '{item}' in prompt:\n{prompt}")

            pos = start + (len(item_ids) - 1 if prefer_last_token else 0)
            positions.append(pos)

        out_positions.append(positions)
    return out_positions

def index_query_key_vecs_last_token(
    cache,
    layer_idx: int,
    num_key_value_groups: int,
):
    """
    Returns:
      q_last: [B, H, D]
      k_all:  [B, H, S, D]
    """
    # cache[...] entries are tuples in your code; [0] is the tensor
    q = cache[query_module_name.format(layer_idx)][0]  # [B, H, S, D] in your setup
    q_last = q[:, :, -1, :]  # [B, H, D]

    k = cache[key_module_name.format(layer_idx)][0]  # [B, H_kv, S, D]
    k_all = repeat_kv(k, num_key_value_groups)  # [B, H, S, D]
    return q_last, k_all


def attn_from_qk(
    q_last: torch.Tensor,  # [B, H, D]
    k_all: torch.Tensor,  # [B, H, S, D]
    attention_mask: torch.Tensor,  # [B, S]  (1 for real tokens, 0 for pad)
) -> torch.Tensor:
    """
    Returns attention probs from the last token query to all keys:
      attn: [B, H, S]
    """
    B, H, S, D = k_all.shape
    scale = 1.0 / math.sqrt(D)
    logits = torch.einsum("bhd,bhsd->bhs", q_last, k_all) * scale  # [B,H,S]

    # mask pads (assumes right padding; if you left-pad, adapt this)
    mask = attention_mask[:, None, :].to(dtype=torch.bool)  # [B,1,S]
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    return torch.softmax(logits, dim=-1)


