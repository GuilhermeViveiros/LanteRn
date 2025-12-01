
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
    judge: LLMJudge
):
    model.eval()
    correct = 0
    incorrect = 0
    invalid = 0
    total = 0
    latent_samples = 0

    for idx, (inputs, labels) in tqdm(enumerate(dataloader), "VisCot Test"):
        # if idx == 1:
        #     break
        # move pixel values to the correct device
        inputs = inputs.to(model.device)
        with torch.no_grad():
            # get gt latent values
            gt_latent_embeds = apply_latent_compression(
                model,
                input_ids=inputs["input_ids"],
                latent_values=inputs.pop("latent_values"),
                latent_grid_thw=inputs.pop("latent_grid_thw"),
            )

            # I'll pass the ground truth latent embeddings to the generate function for debugging purposes
            # this will be removed in the future (just for stress testing purposes)
            output = model.generate(
                **inputs,
                max_new_tokens=124,
                do_sample=False,
                #temperature=0.7,
                #top_p=0.9,
                tokenizer=processor.tokenizer,
                custom_generate=partial(lantern_generate, gt_latent_embeds=None),
                use_cache=False,
            )

            output_ids = output.input_ids
            pred_latent_embeds = output.latent_pred_values            
            decoded_output = processor.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            print(f"idx {idx} decoded_output: {decoded_output}")
            answer = decoded_output.split('<answer>')[-1].split('</answer>')[0].strip()
            
            if pred_latent_embeds is None:
                continue

            latent_samples += 1
            
            mse_loss, sim_loss = compare_latent_embeddings(
                pred_latent_embeds,
                gt_latent_embeds
            )

            #import pdb; pdb.set_trace()
            
            # print(f"mse_loss: {mse_loss} | sim_loss: {sim_loss}")
           
            try:
                result = judge.judge(answer, labels[0])
                logger.info(f"Answer: {answer} | Label: {labels[0]} | Result: {result}")

                total += result

                if result > 0.5:
                    correct += 1
                else:
                    incorrect += 1
            except Exception as e:
                invalid += 1
                logger.info(f"Error judging answer: {e}")

            
            #if (idx+1) % 20 == 0:
            logging.info(f"[{idx+1}] Avg score: {total/(latent_samples):.3f}, Accuracy: ({correct/latent_samples:.3f}), Invalid: ({invalid/latent_samples:.3f}), latent ratio: ({latent_samples/(idx+1):.3f})")
            

    
    total_samples = len(dataloader.dataset)
    result = {
        "avg_score": total/latent_samples if latent_samples > 0 else 0,
        "accuracy": correct/latent_samples if latent_samples > 0 else 0,
        "invalid": invalid/latent_samples if latent_samples > 0 else 0,
        "latent_ratio": latent_samples/total_samples if total_samples > 0 else 0
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
        default="/mnt/scratch-artemis/gviveiros/lantern/checkpoints/model_stage1/checkpoint-5000",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json",
        help="Path to the data"
    )
    args = parser.parse_args()
    
    logging.info('=='*20)
    logging.info('Testing model...')
    logging.info(f"Arguments: {args}")
    logging.info('=='*20)

    # load the model and processor
    model, processor = load_model(model_path=args.model_ref, device_map="cuda", compute_dtype=torch.bfloat16, use_cache=False)  
    assert model.config.vocab_size == 151668, "Embedding size is not correct"

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
    model.config.lvr_sep_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_sep|>")
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    print(f"model.config.lvr_sep_id: {model.config.lvr_sep_id}")
    #model.resize_token_embeddings(len(processor.tokenizer))


    # Load data
    data_module = make_sft_data_module(
        processor=processor,
        data_path=args.data_path,
        generate=True,
        shuffle=False,
        seed=42,
        split_percentages=(0.99, 0.009, 0.001)
    )

    dataset = data_module["test_dataset"]
    logger.info(f"Test dataset size: {len(dataset)}")
    collator = data_module["data_collator"]
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    # load the judge
    judge = LLMJudge(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    # main
    results = viscot_test(model, processor, dataloader, judge)
    # save the results to a json file
    with open(f"results_{args.model_ref.split('/')[-1]}.json", "w") as f:
        json.dump(results, f)
    logging.info(f"Results saved to results_{args.model_ref.split('/')[-1]}.json")
    logging.info(f"Results: {results}")
    logging.info('=='*20)
    logging.info('Testing completed')