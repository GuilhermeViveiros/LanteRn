import csv
import json
from functools import partial
from tqdm import tqdm
import random
import numpy as np
import os
import torch
import argparse
from typing import List
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from src.utils import center_and_crop_image
from evals import run_batch_inference, get_gt_latent_values
from evals.blink_eval import BlinkDataset
from qwen_vl_utils import process_vision_info
from src.utils import extract_mc_answer

class MCDataset(Dataset):
    # mcdataset is a multiple choice dataset that aggregates different datasets into a single one
    def __init__(self, datasets: List[str], use_lvr: bool = True, bbox_ablation: str = None, corrupt_image: bool = False):
        # categories that to be represented in each dataset
        self.data = []
        self.use_lvr = use_lvr
        self.bbox_ablation = bbox_ablation
        self.corrupt_image = corrupt_image
        for dataset in datasets:
            if dataset == "viscot":
                # load the viscot test dataset
                #with open("/mnt/home/gviveiros/LantErn/oe_to_mc.jsonl", "r") as f:
                with open("/e/project1/jureap126/gviveiros/lantern/viscot_mc_test.jsonl", "r") as f:
                    invalid_options = 0
                    for line in f:
                        sample = json.loads(line)
                        sample["dataset"] = "viscot"
                        sample["category"] = "viscot"
                        sample["label"] = sample.pop("answer")

                        # parse options
                        options = sample["options"]
                        # examples
                        #['A. tree branches', 'B. cars', 'C. buildings', 'D. street']
                        #['A. man B. woman C. child D. dog']
                        def parse_options(options):
                            # options should be a nested list of strings
                            if len(options) != 4: # wrong format, return None
                                return None
                            else:
                                # to remove 
                                options = [option.replace(option.split(" ")[0]+" ", "") for option in options]
                                # check if any option is None
                                if any(option is None for option in options):
                                    return None
                                return options

                        options = parse_options(options)
                        if options is None:
                            invalid_options += 1
                            continue


                        
                        sample["question"] = sample["question"] + "\nOptions:\n" + "\n".join([f"{option}" for option in sample["options"]])                        
                        sample["options"] = options
                        # To support distributed training and avoid too many open files,
                        # store image paths here; open images only in __getitem__.
                        img_path = sample["img_path"].replace(
                            "/mnt/data-artemis/gviveiros/lantern/",
                            "/e/project1/jureap126/gviveiros/lantern/",
                        )
                        sample["image"] = [img_path]  # store path, open later
                        self.data.append(sample)
                    print(f"Invalid options: {invalid_options}")
                    
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
        sample = dict(self.data[idx])  # shallow copy to avoid mutating self.data
        if "bbox" in sample:
            cropped_imgs = []
            main_imgs = []
            for img_path, bbox in zip(sample["image"], sample["bbox"]):
                img_clean = Image.open(img_path).convert("RGB")
                img_main = self._blackout_bbox(img_clean, bbox) if self.corrupt_image else img_clean
                # latent crops always use the clean image, matching training setup
                effective_bbox = self._ablate_bbox(bbox, img_clean, self.bbox_ablation)
                cropped_imgs.append(center_and_crop_image(img_clean, effective_bbox))
                main_imgs.append(img_main)
            sample["cropped_images"] = cropped_imgs
            if self.corrupt_image:
                sample["image"] = main_imgs  # replace paths with corrupted PIL images

        return sample

    @staticmethod
    def _blackout_bbox(img: Image.Image, bbox) -> Image.Image:
        """Return a copy of img with the bbox region zeroed out (black)."""
        import numpy as np
        arr = np.array(img)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        arr[y1:y2, x1:x2] = 0
        return Image.fromarray(arr)

    @staticmethod
    def _ablate_bbox(bbox, img: Image.Image, mode: str):
        """Return a (possibly modified) bbox for ablation experiments."""
        if mode is None or mode == "none":
            return bbox
        if mode == "random":
            # Preserve bbox dimensions but place it at a random location in the image.
            x1, y1, x2, y2 = bbox
            W, H = img.width, img.height
            bw, bh = int(x2 - x1), int(y2 - y1)
            new_x1 = random.randint(0, max(0, W - bw))
            new_y1 = random.randint(0, max(0, H - bh))
            return [new_x1, new_y1, new_x1 + bw, new_y1 + bh]
        raise ValueError(f"Unknown bbox_ablation mode: {mode!r}")

#TODO: Improve this... hardcoded just to evaluate monet model
#system_content = "You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. When you decide to generate abstract visual tokens, put them in <|lvr_start|>...<|lvr_end|>. Put your final answer inside <answer> tags"
#system_content = "Put your final answer inside <answer>ANSWER_GOES_HERE</answer> tags."

def collate_fn_mc(samples, processor: AutoProcessor):
    messages, labels, options = [], [], []
    cropped_images = []
    bboxs = []
    for sample in samples:
        labels.append(sample["label"])
        if "options" in sample:
            options.append(sample["options"])
        else:
            options.append(None)
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
    return inputs, labels, options, bboxs, cropped_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Qwen/Qwen2.5-VL-3B-Instruct
    parser.add_argument(
        "--model_ref",
        type=str,
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-3000/"
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1_Monet/checkpoint-890"
        #default="/mnt/scratch-artemis/gviveiros/Monet-SFT-7B/stage3/"
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_8_lambda_0.1/checkpoint-1062/",
    ) 
    parser.add_argument(
        "--lvr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lantern generate or the default generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--bbox_ablation",
        type=str,
        default=None,
        choices=["random"],
        help="Ablation mode for the ground-truth bbox used to build LVR latents. "
             "'random': same bbox size, random location within the image."
    )
    parser.add_argument(
        "--use_gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to replace predicted latent tokens with GT latent embeddings. "
             "Only relevant when --lvr is set."
    )
    parser.add_argument(
        "--corrupt_image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Black out the GT bbox region in the main image before inference. "
             "Use when evaluating a model trained with bbox blackout corruption."
    )
    args = parser.parse_args()
    print("-"*100)
    print(args)
    print("-"*100)

    output_folder = f"{args.output_dir}/mc_results/"
    ablation_suffix = f"_bbox_{args.bbox_ablation}" if args.bbox_ablation else ""
    gt_suffix = "" if args.use_gt else "_no_gt"
    if args.lvr:
        outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}_lvr{gt_suffix}{ablation_suffix}.json"
    else:
        outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}{ablation_suffix}.json"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output file name: {outfile_name}")
    #datasets = ["viscot", "vstar", "blink"]
    datasets = ["viscot"]
    dataset = MCDataset(datasets=datasets, use_lvr=args.lvr, bbox_ablation=args.bbox_ablation, corrupt_image=args.corrupt_image)
    dataset.stats()

    # load the model
    from src.models import load_model
    from src.train import set_latent_tokens
    model, processor = load_model(args.model_ref, device_map="cuda", compute_dtype=torch.bfloat16, use_cache=True)  
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=False)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=False) 
    padding_side='left'   
    processor.tokenizer.padding_side = padding_side
    # check if the latent size is set
    if not hasattr(model.config, "latent_size"):
        logger.info("Warning!!!! Latent size is not set, using 4")
        model.config.latent_size = 4
    
    print(f"model.config.latent_size: {model.config.latent_size}")
    
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    model.config.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")

    processor.tokenizer.padding_side = "left"
    

    # check if the latent size is set
    if args.lvr:
        if not hasattr(model.config, "latent_size"):
            raise ValueError("Warning!!!! Latent size is not set")
        # set_latent_tokens(processor, model, model.config.latent_size)
    

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn_mc, processor=processor))
    bboxs_list = []
    correct_predictions = 0
    latent_ratio = 0
    total_samples = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for step, (batch, labels, options, bboxs, cropped_images) in pbar:
        bboxs_list.extend(bboxs)
        if step >= 300:
            break
        if args.lvr and args.use_gt:
            gt_latent_values, gt_latent_grid_thw = get_gt_latent_values(cropped_images, processor)
            batch["latent_values"] = gt_latent_values
            batch["latent_grid_thw"] = gt_latent_grid_thw
        batch.to(model.device)
        
        # generate ids
        generated_ids = run_batch_inference(model, batch, use_gt=args.use_gt, use_lvr=args.lvr, return_dict=False)
        # generated_ids = output.input_ids if args.lvr else output
        # calculate the latent ratio
        if args.lvr:
            latent_ratio += (generated_ids == model.config.lvr_start_id).any(axis=1).sum().item()

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
        ]
        # decode the generated ids
        decoded_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # TODO: remove this after evaluation        
        # monet model has som bug... workaround stop after the first point
        # maybe its related to the prompt being used..
        #decoded_output = [x.split(".")[0] for x in decoded_output]
        # change observation with answer
        #decoded_output = [x.replace("<observation>", "<answer>").replace("</observation>","</answer>") for x in decoded_output]

        
        print("-"*100)
        print("Options: ", options)
        print("Decoded Output: ", decoded_output)
        print("-"*100)
        # extract the answer from the decoded output
        answers = [extract_mc_answer(x, options) for x, options in zip(decoded_output, options)]
        correct_predictions += np.sum(np.array(answers) == np.array(labels))

        print("Answers: ", answers, "Labels: ", labels, "Correct Predictions: ", correct_predictions)
        total_samples += len(labels)
        current_accuracy = correct_predictions / total_samples
        current_latent_ratio = latent_ratio / total_samples
        pbar.set_description(f"Evaluating | Accuracy: {current_accuracy:.4f}, Latent Ratio: {current_latent_ratio:.4f}")

    # append the results to a csv file with model name and accuracy
    with open(outfile_name, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([args.model_ref, current_accuracy, current_latent_ratio])