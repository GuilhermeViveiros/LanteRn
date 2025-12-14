import os
import json
import re
import numpy as np
import logging
import argparse
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import DataLoader
from src.models import load_model
from datasets import load_dataset
from evals import run_batch_inference
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
# import fuzzy matching
from fuzzywuzzy import fuzz


logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("LantErn-Test-VStar")

# ==== Core utilities ====
def extract_answer(response: str, options: list[str]) -> str:
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    
    if given_answer:
        match = re.search(r"(?:Answer:\s*)?(?:\(|\b)([A-Z])(?:\)|\b)", given_answer)
        if match:
            given_answer = match.group(1)
        else:
            # check using fuzzy matching
            if given_answer is not None:
                scores = [fuzz.ratio(given_answer.lower(), opt.lower()) for opt in options]
            
                max_score = max(scores)
                if max_score > 90:
                    given_answer = ["A", "B", "C", "D", "E", "F"][scores.index(max_score)]
                else:
                    given_answer = None


    return given_answer

def collate_fn(batch, processor: AutoProcessor):
    messages = []
    options = []
    labels = []
    categories = []

    for dat in batch:
        labels.append(dat['label'])
        categories.append(dat['category'])
        # HACK: remove the prefix from vstar
        # this prompt can mislead lantern generation
        question = dat['text'].split("Answer with the option's letter from the given choices directly.")[0].strip()
        opts = question.split("\n")[1:]
        opts = [opt.split(" ")[-1].strip() for opt in opts]
        options.append(opts)
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dat['image']},
                    {"type": "text", "text": question}
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

    return inputs, options, labels, categories


def vstar_eval(
    model, 
    processor,
    dataloader: DataLoader,
    use_lvr: bool
):
    model.eval()
    print("Using LVR: ", use_lvr)
    accuracy = 0 # number of correct answers
    latent_samples = 0 # number of samples with latent tokens generated
    total_samples = 0 # total number of samples
    res_by_category = {}
    for category in dataset.unique("category"):
        res_by_category[category] = {
            "accuracy": 0,
            "latent_ratio": 0,
            "total": 0,
        }

    for step, (inputs, options, labels, categories) in tqdm(enumerate(dataloader), total=len(dataloader), desc="VisCot Test"):
      
        # run batch inference
        generated_ids = run_batch_inference(model, inputs, use_lvr=use_lvr)
        latent_samples += (generated_ids == model.config.lvr_start_id).any(axis=1).sum().item()
        total_samples += len(inputs.input_ids)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # decode the generated ids
        batch_decoded_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"[{step}] Batch decoded output:")
        for i, x in enumerate(batch_decoded_output):
            print(f"[{i}] {x}")
        
        # extract the answer from the decoded output
        answers = [extract_answer(x, options[i]) for i, x in enumerate(batch_decoded_output)]
        if answers[0] is not None and len(answers[0]) == 0:
            import ipdb; ipdb.set_trace()
        # check if the answer is correct
        results = np.array([answers[i] == labels[i] for i in range(len(answers))])
        accuracy += results.sum()
        
        for idx, (res, category) in enumerate(zip(results, categories)):
            res_by_category[category]["total"] += 1
            res_by_category[category]["accuracy"] += res
            res_by_category[category]["latent_ratio"] += (generated_ids[idx] == model.config.lvr_start_id).any(axis=-1).sum().item()
        
        logging.info(f"Answer: {answers} | Label: {labels} | Result: {results} | Accuracy: {accuracy / total_samples} | Latent Ratio: {latent_samples / total_samples}")
    
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
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/lambda_mse_0.2/checkpoint-995",
        help="Path to the model checkpoint"
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
    dataset = load_dataset("lmms-lab/vstar-bench")["test"]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, processor=processor), shuffle=False)

    # evaluate vstar
    results = vstar_eval(model, processor, dataloader, use_lvr=args.lvr)

    
    output_folder = f"{args.output_dir}/vstar/"
    outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}.json"
    os.makedirs(output_folder, exist_ok=True)
    
    # save the results to a json file
    with open(outfile_name, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {outfile_name}")
    logging.info(f"Results: {results}")
    logging.info('=='*20)
    logging.info('Testing completed')