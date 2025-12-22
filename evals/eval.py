import csv
from torch.utils.data import Dataset
from src.utils import center_and_crop_image
from src.lantern_generate.generate import generate as lantern_generate
from evals import run_batch_inference
from evals.blink_eval import BlinkDataset
from functools import partial
import torch
import argparse
from src.utils import extract_mc_answer
from src.models.utils import apply_latent_compression
from typing import List
import json
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import numpy as np

class MCDataset(Dataset):
    # mcdataset is a multiple choice dataset that aggregates different datasets into a single one
    def __init__(self, datasets: List[str], use_lvr: bool = True):
        # categories that to be represented in each dataset
        self.data = []
        self.use_lvr = use_lvr
        for dataset in datasets:
            if dataset == "viscot":
                # load the viscot test dataset
                #with open("/mnt/home/gviveiros/LantErn/oe_to_mc.jsonl", "r") as f:
                with open("/mnt/scratch-artemis/gviveiros/lantern/oe_to_mc/viscot_mc_test.jsonl", "r") as f:
                    for line in f:
                        sample = json.loads(line)
                        sample["dataset"] = "viscot"
                        sample["category"] = "viscot"
                        sample["label"] = sample.pop("answer")
                        sample["question"] = sample["question"] + "\nOptions:\n" + "\n".join([f"{option}" for option in sample.pop("options")])
                        # To support distributed training and avoid too many open files,
                        # store image paths here; open images only in __getitem__.
                        sample["image"] = [sample["img_path"]]  # store path, open later
                        self.data.append(sample)
                    
            elif dataset == "vstar":
                # load the vstar test dataset
                vstar_dataset = load_dataset("lmms-lab/vstar-bench")["test"]
                for sample in vstar_dataset:
                    imgs = sample["image"]
                    # convert img to PIL if it is a string
                    if not isinstance(imgs, list):
                        imgs = [imgs]
                    # append data to the dataset
                    self.data.append({
                        "question": sample["text"],
                        "image": imgs,
                        "label": sample["label"],
                        "category": sample["category"],
                        "dataset": "vstar",
                    })
                
            elif dataset == "blink":
                blink_dataset = BlinkDataset()
                for sample in blink_dataset:
                    imgs = sample["image"]
                    # if not list, convert to list
                    if not isinstance(imgs, list):
                        imgs = [imgs]
                    # append data to the dataset
                    self.data.append({
                        "question": sample["question"],
                        "image": imgs,
                        "label": sample["label"]
                    })
            else:
                raise ValueError(f"Dataset {dataset} not supported")

       
    def stats(self):
        print(f"Dataset size: {len(self.data)}")
        # get the nuber of samples for each dataset
        samples_by_dataset = {}
        for sample in self.data:
            if sample['dataset'] not in samples_by_dataset:
                samples_by_dataset[sample['dataset']] = 0
            samples_by_dataset[sample['dataset']] += 1
        for dataset, count in samples_by_dataset.items():
            print(f"Dataset {dataset} size: {count}")
        # get the number of samples for each category
        samples_by_category = {}
        for sample in self.data:
            if sample['category'] not in samples_by_category:
                samples_by_category[sample['category']] = 0
            samples_by_category[sample['category']] += 1
        for category, count in samples_by_category.items():
            print(f"Category {category} size: {count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # check if the image is a path
        sample = self.data[idx]
        # get cropped img bboxs
        # if not using lvr, modify the prompt for every other model
        #if not self.use_lvr:
        sample["question"] += "\nAnswer with the option's letter from the given choices directly."
        if "bbox" in sample:
            cropped_imgs = [center_and_crop_image(Image.open(img), bbox) for img, bbox in zip(sample["image"], sample["bbox"])]
            sample["cropped_images"] = cropped_imgs
        
        return sample

def collate_fn_mc(samples, processor: AutoProcessor):
    messages, labels = [], []
    cropped_images = []
    bboxs = []
    for sample in samples:
        labels.append(sample["label"])
        bboxs.append(sample["bbox"])
        cropped_images.append(sample["cropped_images"])
        messages.append([
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in sample["image"]],
                    {"type": "text", "text": sample["question"]}
                ]
            }
        ])
    # prepare the inputs
    inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=inputs,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, labels, bboxs, cropped_images

def get_gt_latent_values(cropped_images, processor):
    # run batch inference
    messages = [
        [{
            "role": "assistant",
            "content": [{"type": "image", "image": img[0]}]
        }] for img in cropped_images
    ]
    # apply processor
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    gt_latent_values = inputs["pixel_values"]
    gt_latent_grid_thw = inputs["image_grid_thw"]
    
    return gt_latent_values, gt_latent_grid_thw


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Qwen/Qwen2.5-VL-3B-Instruct
    parser.add_argument(
        "--model_ref",
        type=str,
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.1/checkpoint-1062"
    )
    parser.add_argument(
        "--lvr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lantern generate or the default generation"
    )
    args = parser.parse_args()
    print("-"*100)
    print(args)
    print("-"*100)
    #datasets = ["viscot", "vstar", "blink"]
    datasets = ["viscot"]
    dataset = MCDataset(datasets=datasets, use_lvr=args.lvr)
    dataset.stats()

    # load the model
    from src.models import load_model
    model, processor = load_model(model_path=args.model_ref, device_map="cuda", compute_dtype=torch.bfloat16, use_cache=True)
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=False) 
    processor.tokenizer.padding_side = "left"
    # check if the latent size is set
    if args.lvr:
        if not hasattr(model.config, "latent_size"):
            raise ValueError("Warning!!!! Latent size is not set")
    
    #print(f"model.config.latent_size: {model.config.latent_size}")    

    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=partial(collate_fn_mc, processor=processor))
    bboxs_list = []
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for step, (batch, labels, bboxs, cropped_images) in pbar:
        bboxs_list.extend(bboxs)
        if args.lvr:
            gt_latent_values, gt_latent_grid_thw = get_gt_latent_values(cropped_images, processor)
            batch["latent_values"] = gt_latent_values
            batch["latent_grid_thw"] = gt_latent_grid_thw
            batch.to(model.device)
        # run inference with the gt latent value
        generated_ids = run_batch_inference(model, batch, use_gt=False, use_lvr=args.lvr)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
        ]
        # decode the generated ids
        decoded_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Decoded Output: ", decoded_output)
        # extract the answer from the decoded output
        answers = [extract_mc_answer(x) for x in decoded_output]
        correct_predictions += np.sum(np.array(answers) == np.array(labels))

        print("Answers: ", answers, "Labels: ", labels, "Correct Predictions: ", correct_predictions)
        total_samples += len(labels)
        current_accuracy = correct_predictions / total_samples
        pbar.set_description(f"Evaluating | Accuracy: {current_accuracy:.4f}")

# append the results to a csv file with model name and accuracy
with open("mc_results.csv", "a") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([args.model_ref, current_accuracy])