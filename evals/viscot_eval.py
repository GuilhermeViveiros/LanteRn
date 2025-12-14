
import os
import json
import time
import logging
import argparse
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import DataLoader
from src.models import load_model
from src.datasets.sft_data import make_sft_data_module
from src.lantern_generate.generate import generate as lantern_generate
from src.models.utils import apply_latent_compression
from src.judge import LLMJudge
from evals import run_batch_inference



logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("LantErn-Test")

def compare_latent_embeddings(
    latent_embeds_pred: torch.FloatTensor,
    gt_latent_embeds: torch.FloatTensor,
):
    sim_loss = 1-torch.nn.functional.cosine_similarity(latent_embeds_pred, gt_latent_embeds).mean(axis=-1)
    #mse_loss = torch.nn.functional.mse_loss(latent_embeds_pred, gt_latent_embeds)
    mse_loss = ((latent_embeds_pred - gt_latent_embeds) ** 2).mean(dim=(1, 2))

    return mse_loss, sim_loss

# Load the model and processor
def viscot_test(
    model, 
    processor,
    dataloader: DataLoader,
    judge_name: str,
    use_gt: bool = False,
    use_lvr: bool = True
):
    judge = LLMJudge(model_id=judge_name)
    logger.info(f"Using judge: {judge_name}")
    model.eval()
    correct = 0 # number of correct answers
    invalid = 0 # number of invalid answers (parsing error)
    total = 0 # total score
    average_mse_loss = 0 # average MSE loss
    latent_samples = 0 # number of samples with latent tokens generated
    total_samples = 0 # total number of samples

    # print latent tokens
    logger.info(f"{'Using ground truth latent embeddings.' if use_gt else 'Using predicted latent embeddings.'}")
    logger.info(f"latent tokens: {model.config.additional_special_tokens}")

    for step, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="VisCot Test"):
        # move pixel values to the correct device
        inputs = inputs.to(model.device)
        # run batch inference
        generated_ids = run_batch_inference(
            model,
            inputs,
            use_lvr=use_lvr,
            use_gt=use_gt
        )
        total_samples += len(inputs.input_ids)
        
        # trim the generated ids to the length of the input ids
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        # decode the generated ids
        batch_decoded_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # extract the answer from the decoded output
        batch_parsed_output = [
            x.split('<answer>')[-1].split('</answer>')[0].strip()
            if '<answer>' in x else None
            for x in batch_decoded_output
        ]

        try:
            results = judge.judge(batch_parsed_output, labels)
            logger.info(f"Answer: {batch_parsed_output} | Label: {labels} | Result: {results}")

            total += results.sum()
            correct += (results > 0.5).sum()
            
        except Exception as e:
            invalid += 1
            logger.info(f"Error judging answer: {e}")

            
        logging.info(f"[{step}] \
            Avg score: {total/total_samples:.3f}, \
            Accuracy: ({correct/total_samples:.3f}), \
            Invalid: ({invalid/total_samples:.3f}), \
            latent ratio: ({latent_samples/total_samples:.3f})"
        )
    
    result = {
        "avg_score": total/total_samples,
        "accuracy": correct/total_samples,
        "invalid": invalid/total_samples,
        "latent_ratio": latent_samples/total_samples,
    }
    logging.info(f"\n{'='*40}\n[Final Results]\n{'-'*40}\n"
                 f"Average Score : {result['avg_score']:.3f}\n"
                 f"Accuracy      : {result['accuracy']:.3f}\n"
                 f"Invalid Ratio : {result['invalid']:.3f}\n"
                 f"Latent Ratio  : {result['latent_ratio']:.3f}\n"
                 f"{'='*40}")
    return result
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ref",
        type=str,
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/model_stage1/checkpoint-5000/",
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/lambda_mse/checkpoint-600/",
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/lambda_mse_0.2/checkpoint-700/",
        #default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/sft_mse_lt_4_lambda_0.0/checkpoint-600/",
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
        "--data_path",
        type=str,
        default="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json",
        help="Path to the data"
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

    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True) 
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
    data_module = make_sft_data_module(
        processor=processor,
        data_path=args.data_path,
        generate=True,
        seed=42,
        latent_size=model.config.latent_size,
        split_percentages=(0.9, 0.097, 0.003)
    )

    dataset = data_module["test_dataset"]
    logger.info(f"Test dataset size: {len(dataset)}")
    collator = data_module["data_collator"]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    # main test function
    results = viscot_test(model, processor, dataloader, judge_name="Qwen/Qwen2.5-VL-3B-Instruct", use_gt=args.use_gt, use_lvr=args.lvr)
    
    output_folder = f"{args.output_dir}/viscot/"
    outfile_name = f"{output_folder}/{'_'.join(args.model_ref.split('/')[-2:])}.json"
    os.makedirs(output_folder, exist_ok=True)
    
    # save the results to a json file
    with open(outfile_name, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {outfile_name}")
    logging.info(f"Results: {results}")
    logging.info('=='*20)
    logging.info('Testing completed')