import re
from typing import Any, Dict, List, Union, Callable

from src.rl.utils import extract_last_answer_from_text


# builder: (model, **kwargs) -> reward_fn
REWARD_REGISTRY: Dict[str, Callable[..., Callable]] = {}


def register_reward(name: str):
    def deco(builder: Callable[..., Callable]) -> Callable[..., Callable]:
        if name in REWARD_REGISTRY:
            raise ValueError(f"Duplicate reward registration: {name}")
        REWARD_REGISTRY[name] = builder
        return builder

    return deco


@register_reward("accuracy")
def build_accuracy_reward(model, **_kwargs):
    # model is always provided; unused here
    return accuracy_reward


@register_reward("structure")
def build_structure_reward(model, **_kwargs):
    return make_structure_reward_from_ids(model)


# ----------------------------
# Reward functions
# ----------------------------


def accuracy_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Reason:
      - Reward is based ONLY on what is inside the last <answer>...</answer>.

    Assumptions:
      - if no <answer> => reward 0
    """
    out = []
    for comp, gt in zip(completions, ground_truth):
        # print(comp)
        text = completion_to_text(comp)
        pred = extract_last_answer_from_text(text)
        if pred == "":
            out.append(0.0)
        else:
            out.append(1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0)
    return out


def make_structure_reward_from_ids(model, **kwargs):
    # Exact contiguous block in IDs
    lvr_block = (
        [model.config.lvr_start_id]
        + [model.config.lvr_sep_id] * model.config.latent_size
        + [model.config.lvr_end_id]
    )

    def _find_subseq(seq, pat):
        L = len(pat)
        if L == 0 or len(seq) < L:
            return False
        for i in range(len(seq) - L + 1):
            if seq[i : i + L] == pat:
                return True
        return False

    def structure_reward(*, completions_ids=None, **kwargs):
        # TRL GRPO uses `completions_ids`; keep fallback just in case
        import pdb; pdb.set_trace()
        if completions_ids is None:
            completions_ids = kwargs.get("completion_ids")
        if completions_ids is None:
            raise ValueError(
                "Reward func didn't receive completions_ids/completion_ids."
            )

        return [1.0 if _find_subseq(ids, lvr_block) else 0.0 for ids in completions_ids]

    return structure_reward


# ----------------------------
# Utils
# ----------------------------


def normalize_answer(s: str) -> str:
    """
    Reason:
      - Simple normalization to reduce noise. TODO: improve?

    Assumptions:
      - Case-insensitive is desired.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _blocks_to_text(content_blocks: Any) -> str:
    if isinstance(content_blocks, str):
        return content_blocks
    if isinstance(content_blocks, list):
        out = []
        for b in content_blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "".join(out)
    return str(content_blocks)


def completion_to_text(
    completion: Union[str, Dict[str, Any], List[Dict[str, Any]]],
) -> str:
    """
    Reason:
      - In TRL's chat/multimodal mode, completions are delivered as chat messages:
        [{'role': 'assistant', 'content': ...}]
      - Sometimes content is a string, sometimes it's list-of-blocks.

    Assumptions:
      - The assistant output is in the last message.
      - If content is blocks, we join the text blocks.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion

    # dict message
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else _blocks_to_text(content)

    # list of messages
    if isinstance(completion, list) and completion:
        last = completion[-1]
        content = last.get("content", "")
        return content if isinstance(content, str) else _blocks_to_text(content)

    return ""
