import re
from typing import Any, Dict, List, Union, Callable
from fuzzywuzzy import fuzz
from src.rl.utils import (
    extract_last_answer_from_text,
    ANSWER_RE,
    THINK_RE
)


# builder: (model, **kwargs) -> reward_fn
REWARD_REGISTRY: Dict[str, Callable[..., Callable]] = {}


def register_reward(name: str):
    def deco(builder: Callable[..., Callable]) -> Callable[..., Callable]:
        if name in REWARD_REGISTRY:
            raise ValueError(f"Duplicate reward registration: {name}")
        REWARD_REGISTRY[name] = builder
        return builder

    return deco


# ----------------------------
# Reward functions
# ----------------------------
@register_reward("accuracy")
def accuracy_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Reason:
      - Reward is based ONLY on what is inside the last <answer>...</answer>.

    Assumptions:
      - if no <answer> => reward 0
    """
    out = []
    for text, gt in zip(completions, ground_truth):
        pred = extract_last_answer_from_text(text)
        if pred == "":
            #print("Pred: ", pred, "GT: ", gt)
            #print("-"*30)
            out.append(0.0)
        else:
            ratio = fuzz.ratio(normalize_answer(pred), normalize_answer(gt))
            #print("Pred: ", pred, "GT: ", gt, "Ratio: ", ratio)
            ratio = 0 if ratio < 70 else ratio / 100.0
            #print("-"*30)
            out.append(ratio)
    return out


@register_reward("structure")
def structure_reward(
    completions: List[str],
    latent_size: int,
    **kwargs
    ) -> List[float]:

    """
    Reason:
      - Reward is based on the structure of the completion.
      - The structure should be: <think>...<lvr_start>...<lvr_end>...</think><answer>...<answer>...</answer>
    Assumptions:
      - if no <think> or no <answer> => reward 0
      - if <think> and <answer> are present but <lvr_start>...</lvr_end> is not => reward 0.5

    """
    # Exact contiguous block in IDs
    lvr_block = (
        "<|lvr_start|>"
        + "<|lvr_sep|>" * latent_size
        + "<|lvr_end|>"
    )

    def _find_subseq(seq: str, pat: str) -> int:
        """
        Check how many times a contiguous block of LVR tokens appears in a sequence.
        """
        pattern = re.escape(lvr_block)
        return len(re.findall(pattern, seq))

    # each sequence should have exactly one answer tag and >= 1 lvr tags
    rewards = []
    for seq in completions:
        answer_tags = ANSWER_RE.findall(seq or "")
        think_tags = THINK_RE.findall(seq or "")
        lvr_tags = _find_subseq(seq, lvr_block)
        
        # if no answer tag return 0
        if len(answer_tags) == 0 or len(answer_tags) > 1 or len(think_tags) == 0 or len(think_tags) > 1:
            rewards.append(0.0)
        # if answer, think and lvr are present add 1
        elif len(answer_tags) == 1 and len(think_tags) == 1 and lvr_tags >= 1:
            rewards.append(1.0)
        # if answer and think are present but lvr is not return 0.5
        elif len(answer_tags) == 1 and len(think_tags) == 1 and lvr_tags == 0:
            rewards.append(0.5)
        
        else:
            rewards.append(0.0)
    return rewards


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
