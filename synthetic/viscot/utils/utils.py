import numpy as np
from typing import List, Tuple
from Levenshtein import ratio
from PIL import Image
import re

def similarity_score(x: str, y: str):
    """
    Calculate the similarity score between two strings.
    Args:
        x: The first string.
        y: The second string.
    Returns:
        The similarity score between the two strings.
    """
    return ratio(x, y) * 100

def parse_output(x: str):
    """
    Parse the output to the correct format.
    The structure should follow something like:

    <think>
    Optional sentence
    <Visual Reasoning></Visual Reasoning>
    Sentence 
    <think>
    <answer> text </answer>
    Args:
        x: The output of the model.
    Returns:
        # we will return a dictionary with the think before <Visual Reasoning>, think before 
        {        
            "pre_visual_text_think": string, # all the thinking that is before the first <Visual Reasoning>
            "post_visual_text_think": List[string], # all the thinking that between the first and last </Visual Reasoning>
            "text_think": string, # all the thinking that is between think when there is no <Visual Reasoning>
            "answer": string # the answer
        }
    """
    
    outputs = {
        "pre_visual_text_think": None,
        "post_visual_text_think": [],
        "text_think": None,
        "answer": None
    }
    
    x = x.strip()
    # replace any occurrence of <tool_call> with <think>
    x = re.sub(r'<tool_call>', '<think>', x)
    # if <think> is not present add it at the beginning of the text
    if '<think>' not in x:
        x = '<think>' + x
    # search for <answer> and </answer>
    answer_start = x.find('<answer>')
    answer_end = x.find('</answer>')
    assert answer_start != -1 and answer_end != -1, "Could not find <answer> and </answer>"
    assert answer_start < answer_end, "Could not find <answer> and then </answer>"

    # seearch for <think> and </think>
    think_start = x.find('<think>')
    think_end = x.find('</think>')
    assert think_start != -1 and think_end != -1, "Could not find <think> and </think>"
    assert think_start < think_end, "Could not find <think> and then </think>"

    # search for <Visual Reasoning> and </Visual Reasoning>
    visual_reasoning_start = x.find('<Visual Reasoning>')
    visual_reasoning_end = x.find('</Visual Reasoning>')

    if visual_reasoning_start == -1:
        # no <Visual Reasoning> found, return the think and answer
        outputs["text_think"] = x[think_start+len('<think>'):think_end].strip()
        outputs["answer"] = x[answer_start+len('<answer>'):answer_end].strip()
        return outputs

    # get how many <Visual Reasoning> are there
    visual_reasoning_count = x.count('<Visual Reasoning>')

    # text between <think> and the first <Visual Reasoning>
    pre_visual_reasoning_text = x[think_start+len('<think>'):visual_reasoning_start].strip()
    start_idx = visual_reasoning_end + len('</Visual Reasoning>')

    # if there are multiple <Visual Reasoning>, extract the text between the first and last </Visual Reasoning>
    for i in range(1,visual_reasoning_count):
        # find the next <Visual Reasoning>
        next_visual_reasoning_start = x.find('<Visual Reasoning>', start_idx)
        # text between the last </Visual Reasoning> and the next <Visual Reasoning>
        post_visual_reasoning_text = x[start_idx:next_visual_reasoning_start].strip()
        outputs["post_visual_text_think"].append(post_visual_reasoning_text)
        start_idx = x.find('</Visual Reasoning>', start_idx) + len('</Visual Reasoning>')
    
    # now extract the final post_visual_reasoning_text between the last </Visual Reasoning> and <answer>
    post_visual_reasoning_text = x[start_idx:think_end].strip()
    outputs["pre_visual_text_think"] = pre_visual_reasoning_text
    outputs["post_visual_text_think"].append(post_visual_reasoning_text)


    # curation of similarity score between the pre_visual_reasoning_text and the post_visual_reasoning_text
    for pre_visual_text, post_visual_text in zip(outputs["pre_visual_text_think"], outputs["post_visual_text_think"]):
        similarity_score_value = similarity_score(pre_visual_text, post_visual_text)
        if similarity_score_value > 60:
            print(f"\033[91mSimilarity score between {pre_visual_text} and {post_visual_text} is {similarity_score}\033[0m")
            # remove the pre_visual_text
            outputs["pre_visual_text_think"] = None
            break

    # extract the answer
    outputs["answer"] = x[answer_start+len('<answer>'):answer_end].strip()
    return outputs

def center_and_crop_image(
    img: Image.Image,
    bbox: List[float],
    output_shape: Tuple[int, int] = None,
    context_scale: float = 1.2
) -> Image.Image:
    """
    Crop an image around a bounding box while preserving maximum resolution.

    Args:
        img: Original image (H, W, C).
        bbox: Bounding box [x1, y1, x2, y2].
        output_shape: Optional (height, width). If None, keep native cropped resolution.
        context_scale: Multiplier for adding context around bbox.

    Returns:
        Cropped (and optionally resized) image, and the transformation matrix.
    """
    H, W = img.height, img.width
    x1, y1, x2, y2 = bbox

    # Compute bbox center and size
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0

    # Apply context scaling
    w *= context_scale
    h *= context_scale

    # Compute new coordinates
    left = max(0, int(cx - w / 2.0))
    right = min(W, int(cx + w / 2.0))
    top = max(0, int(cy - h / 2.0))
    bottom = min(H, int(cy + h / 2.0))

    # Crop directly at native resolution
    cropped = img.crop((left, top, right, bottom))

    # Only resize if user explicitly wants an output shape
    if output_shape is not None:
        cropped = cropped.resize(output_shape)
    # save cropped image
    #cropped.save("img_bbox_0.jpg")
    return cropped


if __name__ == "__main__":
    x = """<think>
    <Visual Reasoning></Visual Reasoning>as
    dasdas
    </think>
    <answer>The children are engaged in a group activity, possibly a storytelling session or a guided tour.</answer><|im_end|>
    """
    for k,v in parse_output(x).items():
        print("--------------------------------")
        print("key: ", k)
        print(v)
        print("--------------------------------")

    print(similarity_score("girl in yellow shirt", "what about the girl in yellow shirt? asdjkloasjkldas das daosindas das"))
   