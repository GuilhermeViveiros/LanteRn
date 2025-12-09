
import os
import json
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
    sim_loss = 1-torch.nn.functional.cosine_similarity(latent_embeds_pred, gt_latent_embeds).mean()
    mse_loss = torch.nn.functional.mse_loss(latent_embeds_pred, gt_latent_embeds)
    return mse_loss, sim_loss

# Load the model and processor
def viscot_test(
    model, 
    processor,
    dataloader: DataLoader,
    judge_name: str,
    use_gt: bool = False
):
    judge = LLMJudge(model_id=judge_name)
    logger.info(f"Using judge: {judge_name}")
    model.eval()
    correct = 0 # number of correct answers
    invalid = 0 # number of invalid answers (parsing error)
    total = 0 # total score
    average_mse_loss = 0 # average MSE loss
    latent_samples = 0 # number of samples with latent tokens generated

    # print latent tokens
    logger.info(f"{'Using ground truth latent embeddings.' if use_gt else 'Using predicted latent embeddings.'}")
    logger.info(f"latent tokens: {model.config.additional_special_tokens}")

    for step, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="VisCot Test"):
        step += 1
        # move pixel values to the correct device
        inputs = inputs.to(model.device)
        with torch.no_grad():
            if "latent_values" not in inputs:
                continue
            # get gt latent values
            gt_latent_embeds = apply_latent_compression(
                model,
                input_ids=inputs["input_ids"],
                latent_values=inputs.pop("latent_values"),
                latent_grid_thw=inputs.pop("latent_grid_thw"),
                latent_size=model.config.latent_size,
            )
            # for now I just support one sample at a time
            assert inputs["input_ids"].shape[0] == 1, "Only batch size of 1 is supported"

            # I'll pass the ground truth latent embeddings to the generate function for debugging purposes
            # this will be removed in the future (just for stress testing purposes)
            output = model.generate(
                **inputs,
                max_new_tokens=124,
                do_sample=False,    
                tokenizer=processor.tokenizer,
                custom_generate=partial(lantern_generate, gt_latent_embeds=gt_latent_embeds if use_gt else None),
                use_cache=False,
            )

            output_ids = output.input_ids
            pred_latent_embeds = output.latent_pred_values            
            decoded_output = processor.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


            # check if <answer> is present
            # if '<answer>' not in decoded_output:
            #     invalid += 1
            #     continue
            
            answer = decoded_output.split('<answer>')[-1].split('</answer>')[0].strip()
            
            if pred_latent_embeds is not None:
                latent_samples += 1
                mse_loss, _ = compare_latent_embeddings(
                    pred_latent_embeds,
                    gt_latent_embeds
                )
                average_mse_loss += mse_loss

            try:
                result = judge.judge(answer, labels[0])
                logger.info(f"Answer: {answer} | Label: {labels[0]} | Result: {result}")

                total += result

                if result > 0.5:
                    correct += 1
                
            except Exception as e:
                invalid += 1
                logger.info(f"Error judging answer: {e}")

            logging.info(f"[{step}] \
                Avg score: {total/step:.3f}, \
                Accuracy: ({correct/step:.3f}), \
                Invalid: ({invalid/step:.3f}), \
                latent ratio: ({latent_samples/step:.3f}), \
                average MSE loss: {average_mse_loss/step:.3f}"
            )
            
    result = {
        "avg_score": total/step,
        "accuracy": correct/step,
        "invalid": invalid/step,
        "latent_ratio": latent_samples/step,
        "average_mse_loss": average_mse_loss/step
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
        help="Path to the model checkpoint"
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
    args = parser.parse_args()
    
    logging.info('=='*20)
    logging.info('Testing model...')
    logging.info(f"Arguments: {args}")
    logging.info('=='*20)

    os.makedirs(args.output_dir, exist_ok=True)

    # load the model and processor
    model, processor = load_model(model_path=args.model_ref, device_map="cuda", compute_dtype=torch.bfloat16, use_cache=False)  
    #model, processor = load_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device_map="cuda", compute_dtype=torch.bfloat16, use_cache=False)  


    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_sep|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)
    model.config.additional_special_tokens = [
        "<|lvr_start|>",
        "<|lvr_sep|>",
        "<|lvr_end|>"
    ]
    
    padding_side='left'   
    processor.tokenizer.padding_side = padding_side
    # check if the latent size is set
    if model.config.latent_size is None:
        logger.info("Warning!!!! Latent size is not set, using 4")
        model.config.latent_size = 4
    
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    model.config.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    # assert model.config.vocab_size == 151668, "Embedding size is not correct"
    # assert model.config.lvr_sep_id == processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    # assert model.config.lvr_start_id == processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    # assert model.config.lvr_end_id == processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    # print(f"model.config.lvr_sep_id: {model.config.lvr_sep_id}")
    #model.resize_token_embeddings(len(processor.tokenizer))


    # Load data
    data_module = make_sft_data_module(
        processor=processor,
        data_path=args.data_path,
        generate=True,
        seed=42,
        split_percentages=(0.9, 0.0905, 0.005)
        latent_size=model.config.latent_size,
        split_percentages=(0.9, 0.097, 0.003)
    )

    dataset = data_module["test_dataset"]
    logger.info(f"Test dataset size: {len(dataset)}")
    collator = data_module["data_collator"]
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    # main
    results = viscot_test(model, processor, dataloader, judge_name="Qwen/Qwen2.5-VL-7B-Instruct", use_gt=False)
    # save the results to a json file
    with open(f"{args.output_dir}/results_{args.model_ref.split('/')[-1]}.json", "w") as f:
        json.dump(results, f)
    logging.info(f"Results saved to {args.output_dir}/results_{args.model_ref.split('/')[-1]}.json")
    logging.info(f"Results: {results}")
    logging.info('=='*20)
    logging.info('Testing completed')