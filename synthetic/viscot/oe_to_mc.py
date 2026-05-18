"""
Dataset for supervised fine-tuning (SFT)
"""
import json

# import logger
import logging
import os
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

logger = logging.getLogger("LantErn-OE-to-MC")

def center_and_crop_image(
    img: Image.Image,
    bbox: list[float],
    output_shape: tuple[int, int] = None,
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

    cropped.parent_filename = img.filename

    # save cropped image
    #cropped.save("img_bbox_0.jpg")
    return cropped


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dummy: bool = False,
        latent_size: int = 4,
    ):
        super().__init__()
        self.latent_size = latent_size
        with open(data_path) as f:
            self.dataset = json.load(f)
        # remove sample textvqa/34084d4c3c347b83.jpg
        for data in self.dataset: # MINOR BUGG: ignore this sample for now
            if data["img_path"] == "/mnt/scratch-artemis/gviveiros/lantern/textvqa/34084d4c3c347b83.jpg":
                self.dataset.remove(data)

        def pre_validation(data, idx):
            # ignore samples with more than 1 bbox
            if len(data["bboxs"]) > 1:
                return False
            return True


        logger.info(f"Number of examples of VisCoT data: {len(self.dataset)}")
        # remove cases where the image is too large and bboxs are more than 1
        self.dataset = [data for data, idx in zip(self.dataset, range(len(self.dataset))) if pre_validation(data, idx)]
        logger.info(f"Number of examples of VisCoT data after removing examples with more than 1 bbox: {len(self.dataset)}")
        #self.dataset = [data for data, idx in zip(self.dataset, range(len(self.dataset))) if filter_too_large_images(data, idx)]
        #logger.info(f"Number of examples of VisCoT data after removing examples with too large images: {len(self.dataset)}")

        # if dummy, we only use the first 1000 examples
        if dummy:
            #import random
            #self.dataset = random.sample(self.dataset, min(5000, len(self.dataset)))
            self.dataset = self.dataset[:1000]

        #self.dataset = self.dataset[:100]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # retrieve the image
        data = self.dataset[idx]
        # Extract data fields
        question = data["question"]
        reasoning_traces = data["reasoning_traces"]
        answer = reasoning_traces["answer"]
        pre_visual_latent_reasoning = reasoning_traces.get("pre_visual_text_think", None)
        post_visual_latent_reasoning = reasoning_traces.get("post_visual_text_think", None)
        text_only_reasoning = reasoning_traces.get("text_think", None)

        # Validate reasoning traces
        if text_only_reasoning is None:
            assert post_visual_latent_reasoning is not None or pre_visual_latent_reasoning is not None, \
                "If text_reasoning is not None, post_visual_latent_reasoning or pre_visual_latent_reasoning must be not None"

        # Extract the image and process bboxes
        # img = Image.open(data["img_path"])

        return {
            "question": question,
            "answer": data["answer"],
            "img_path": data["img_path"],
            "bboxs": data["bboxs"],
        }


def collate_fn_sft(samples: list[dict], processor: AutoProcessor):
    SYSTEM_PROMPT = (
    "You're a helpful assistant. Your job is to convert open ended questions into "
    "multiple choice questions.\n"
    "For each sample, you will be given the question, the answer and the associated image.\n"
    "You will need to convert the open ended question into a multiple choice question.\n"
    "You will need to generate 4 multiple choice options for the question and the correct answer, using A, B, C, D as the options.\n"
    "Add the multiple choice options inside <options> and the correct answer inside <answer>, using A, B, C, D as the options."
    )

    questions = []
    answers = []
    img_paths = []
    messages = []
    bboxs = []
    for sample in samples:
        questions.append(sample["question"])
        answers.append(sample["answer"])
        img_paths.append(sample["img_path"])
        bboxs.append(sample["bboxs"])
        messages.append([
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["img_path"]},
                {"type": "text", "text": "Question = " + sample["question"]},
                {"type": "text", "text": "Answer = " + sample["answer"]}
            ]},
        ])

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt"
    )

    return inputs, questions, answers, bboxs, img_paths
if __name__ == "__main__":
    # load the visual model
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=3500 * 28 * 28,
    )
    processor.padding_side = "left"
    processor.tokenizer.padding_side = "left"

    from tqdm import tqdm
    data_path="/mnt/scratch-artemis/gviveiros/lantern/LantErn_VisCot_data.json"
    sft_dataset = SFTDataset(data_path, latent_size=4)
    seed = 42
    split_percentages = (0.9, 0.097, 0.003)
    train_percentage, \
        eval_percentage, \
            test_percentage = split_percentages

    train_size = int(train_percentage * len(sft_dataset))
    eval_size = int(eval_percentage * len(sft_dataset))
    test_size = int(test_percentage * len(sft_dataset))

    if test_size+eval_size+train_size < len(sft_dataset):
        eval_size += len(sft_dataset) - (test_size+eval_size+train_size) # add the remaining samples to the test_size

    logger.info(f"Total size: {len(sft_dataset)}, train: {train_size}, eval: {eval_size}, test: {test_size}")

    from torch.utils.data import random_split
    _, eval_dataset, test_dataset = random_split(sft_dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(seed))

    # add subsets
    dataset = test_dataset

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=partial(collate_fn_sft, processor=processor))

    json_output = []
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )

    # check the current sample saved and start from the next sample
    idx = 0
    if os.path.exists("oe_to_mc.jsonl"):
        with open("oe_to_mc.jsonl") as f:
            for line in f:
                idx += 1
                #current_sample = json.loads(line)
    print(f"Current sample idx: {idx}")
    step = 0
    for _, (inputs, questions, answers, bboxs, img_paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if step < idx:
            continue
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for question, answer, img_path, bbox, output_text in zip(questions, answers, img_paths, bboxs, output_text):
            # extract the options and the correct answer from the output text
            try:
                options = output_text.split("<options>")[1].split("</options>")[0].split("\n")
                options = [option.strip() for option in options if option.strip() if option.strip() != ""]
                correct_answer = output_text.split("<answer>")[1].split("</answer>")[0]
                #print(f"Question: {question}", f"Options: {options}", f"Correct Answer: {correct_answer}", f"Image Path: {img_path}", "-"*100)
                json_output.append({
                    "question": question,
                    "options": options,
                    "answer": correct_answer,
                    "img_path": img_path,
                    "bbox": bbox,
                })
            except Exception as e:
                print(f"Error processing sample {step}: {e}")
                continue
        step += len(questions)

        # save progress every 500 samples
        if step % 200 == 0:
            with open("oe_to_mc.jsonl", "a") as f:
                for item in json_output:
                    f.write(json.dumps(item) + "\n")
            #json_output = []
            print(f"Saved {len(json_output)} samples to oe_to_mc.jsonl")
            json_output = []
        #break
# save the json output
print(f"Saved {len(json_output)} samples to oe_to_mc.jsonl")
