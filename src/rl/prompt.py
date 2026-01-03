from typing import Any, Dict, List, Optional, Union


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


def make_prompt_messages(user_text: str, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reason:
      - This is the TRL multimodal chat format.
      - IMPORTANT for Arrow: keep structure consistent. So every content item is a dict
        that always has keys 'type' and 'text' (even for image block).

    Assumptions:
      - Exactly one image per example.
      - TRL pairs the dataset's 'images' list with the {"type":"image"} block(s).
    """
    prompt = {
        "role": "user",
        "content": [
            {"type": "image", "text": ""},  # keep 'text' key to avoid schema quirks
            {"type": "text", "text": user_text},
        ],
    }
    if system_prompt is not None:
        raise ValueError("System prompt is not supported yet")
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}, 
            prompt
        ]
    else:
        return [prompt]


def build_system_prompt(latent_size: int) -> str:
    lvr_block = "<|lvr_start|>" + "<|lvr_sep|>" * latent_size + "<|lvr_end|>"
    return (
        "STRICT OUTPUT FORMAT.\n"
        "You must output EXACTLY TWO things, in this exact order:\n"
        f"1) {lvr_block}\n"
        "2) <answer>ANSWER_TEXT</answer>\n"
        "\n"
        "Rules:\n"
        "- Be extremely brief.\n"
        "- Do NOT output anything else (no explanations, no extra words, no extra tags).\n"
        "- Output EXACTLY ONE contiguous LVR block, exactly as shown above, with no spaces.\n"
        "- The LVR block must be OUTSIDE the <answer>...</answer> tags.\n"
        "- Output EXACTLY ONE <answer>...</answer> block.\n"
        "- Inside <answer>...</answer>, output ONLY the final answer text.\n"
        "- The <answer>...</answer> block must be the FINAL characters of your response (no trailing text).\n"
        "\n"
        f"Correct example:\n{lvr_block}<answer>No</answer>\n"
        "Incorrect examples:\n"
        "- Any extra text before/after.\n"
        "- Multiple <answer> blocks.\n"
        "- LVR block inside <answer>.\n"
    )
