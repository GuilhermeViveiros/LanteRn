
import os
import json
import time
import numpy as np
import logging
import argparse
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from src.models import load_model
from src.datasets.sft_data import make_sft_data_module
from src.lantern_generate.generate import generate as lantern_generate
from src.models.utils import apply_latent_compression
from datasets import load_dataset
from evals import run_batch_inference
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils import extract_mc_answer



logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("LantErn-Test-Blink")

BLINK_CATEGORIES =  ['Object_Localization', 'Spatial_Relation'] #'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']

class BlinkDataset(Dataset):
    def __init__(self, categories=BLINK_CATEGORIES):
        self.categories = categories
        self.processed_data = self.load_dataset()

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def load_dataset(self):
        processed_data = []
        for category in BLINK_CATEGORIES:
            blink_dataset = load_dataset("BLINK-Benchmark/BLINK", category)['val']
            # iterate each sample
            for dat in blink_dataset:
                idx = dat["idx"]
                choices = dat["choices"]
                letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]

                #letters = string.ascii_uppercase 
                paired = list(zip(letters, choices))
                option_string = ""
                options = []
                for letter, choice in paired:
                    option_string += f"{letter}. {choice}\n"
                    options.append(f"{letter}. {choice}")
                if len(dat['answer']) >1:
                    ans = dat['answer'][1].upper()
                else:
                    ans = dat['answer'][0].upper()
                images = []
                for k in ['image_1','image_2','image_3','image_4']:
                    if k in dat and dat[k] is not None:
                        images.append(dat[k])
                question = dat['question'] + "\nOptions:\n" + option_string
                buffer = {
                    "question_id": idx,
                    "image": images,
                    "query": question,
                    "label": ans,
                    "category": category,
                    "options": options
                }
                processed_data.append(buffer)
        
        return processed_data


def collate_fn(batch, processor: AutoProcessor):
    messages = []
    labels = []
    categories = []
    options = []
    for dat in batch:
        labels.append(dat['label'])
        categories.append(dat['category'])
        options.append(dat['options'])
        messages.append([
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in dat['image']],
                    {"type": "text", "text": dat['query']}
                ]
            }
        ])
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs, labels, categories, options


def blink_eval(
    model, 
    processor,
    dataloader: DataLoader,
    use_lvr: bool
):
    model.eval()
    accuracy = 0 # number of correct answers
    latent_samples = 0 # number of samples with latent tokens generated
    total_samples = 0 # total number of samples
    res_by_category = {}
    for category in BLINK_CATEGORIES:
        res_by_category[category] = {
            "accuracy": 0,
            "latent_ratio": 0,
            "total": 0,
        }

    for step, (inputs, labels, categories, options) in tqdm(enumerate(dataloader), total=len(dataloader), desc="VisCot Test"):
        # run batch inference
        outputs = run_batch_inference(model, inputs, use_lvr=use_lvr)
        generated_ids = outputs.input_ids if use_lvr else outputs
        latent_samples += (generated_ids == model.config.lvr_start_id).any(axis=1).sum().item()
        total_samples += len(inputs.input_ids)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # decode the generated ids
        batch_decoded_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # extract the answer from the decoded output
        answers = [extract_mc_answer(x, options) for x, options in zip(batch_decoded_output, options)]
        results = np.array([answers[i] == labels[i] for i in range(len(answers))])

        for idx, (res, category) in enumerate(zip(results, categories)):
            res_by_category[category]["accuracy"] += res
            res_by_category[category]["latent_ratio"] += (generated_ids[idx] == model.config.lvr_start_id).sum().item()
            res_by_category[category]["total"] += 1

        logger.info(f"[{step}] Answer: {answers} | Label: {labels} | Result: {results}")
        
        
        accuracy += results.sum()
            
        logging.info(f"[{step}] \
            Accuracy: ({accuracy/total_samples:.3f}), \
            latent ratio: ({latent_samples/total_samples:.3f})"
        )

    for category in res_by_category.keys():
        res_by_category[category]["accuracy"] /= res_by_category[category]["total"]
        res_by_category[category]["latent_ratio"] /= res_by_category[category]["total"]
        logging.info(f"Category: {category} | Accuracy: {res_by_category[category]['accuracy']:.3f} | Latent Ratio: {res_by_category[category]['latent_ratio']:.3f}")

    return res_by_category
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ref",
        type=str,
        default="/mnt/scratch-hades/nunogoncalves/LantErn/checkpoints-rl/checkpoint-16153",
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/grpo_lt_8_lambda_0.1/checkpoint-500",
        help="Path to the model checkpoint"
    )

    parser.add_argument(
        "--use_gt",
        type=bool,
        default=False,
        help="Whether to use ground truth latent embeddings"
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
        "--lvr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lantern generate or the default generation"
    )

    args = parser.parse_args()
    
    logging.info('=='*20)
    logging.info('Testing model...')
    logging.info(f"Arguments: {args}")
    logging.info('=='*20)

    os.makedirs(args.output_dir, exist_ok=True)

    # load the model and processor
    model, processor = load_model(model_path=args.model_ref, device_map="cuda", compute_dtype=torch.bfloat16, use_cache=True)  
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


    # Load data
    dataset = BlinkDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, processor=processor), shuffle=False)

    # evaluate blink
    results = blink_eval(model, processor, dataloader, use_lvr=args.lvr)
    
    output_folder = f"{args.output_dir}/blink/"
    if args.lvr:
        outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}_lvr.json"
    else:
        outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}.json"
    os.makedirs(output_folder, exist_ok=True)
    
    # save the results to a json file
    with open(outfile_name, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {outfile_name}")
    logging.info(f"Results: {results}")
    logging.info('=='*20)