import json
import os
from functools import partial

from tqdm import tqdm

from utils.context import system_prompt
from utils.inference.transformer_inference import batch_inference as batch_inference_transformer
from utils.inference.transformer_inference import load_model as load_model_transformer
from utils.inference.vllm_inference import batch_inference as batch_inference_vllm
from utils.inference.vllm_inference import load_model as load_model_vllm
from utils.prepare_vcot_data import VCoTData
from utils.utils import parse_output


def download_data():
    import os

    from huggingface_hub import hf_hub_download

    # 1. Configuration
    REPO_ID = "deepcs233/Visual-CoT"  # the dataset repo id on HF
    REPO_TYPE = "dataset"
    LOCAL_DIR = "/mnt/data-artemis/gviveiros/LVR-Finetune/visual_cot_images"  # local folder to download into
    TAR_FILES = [
        "cot_images_tar_split/cot_images_00",
        "cot_images_tar_split/cot_images_01",
        # add all the tar files you need …
    ]

    # 2. Create local dir
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # 3. Download each tar file
    for fname in TAR_FILES:
        print(f"Downloading {fname} …")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=fname,
            local_dir=LOCAL_DIR,
            use_auth_token=None,  # set token if private
        )
        print(f"Downloaded to {local_path}")


def create_synthetic_data(dataloader, inference, output_json_path, save_every=5):
    print("=" * 80)
    print("🚀 Initiating synthetic data creation process...")
    print("=" * 80)
    out = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            outputs = inference(batch, verbose=False)
        except Exception as e:
            print(f"\033[91mError in batch {batch_idx}: {e}\033[0m")
            continue
        try:
            parsed_outputs = [parse_output(output) for output in outputs]
        except Exception as e:
            print(f"\033[91mError following structured reasoning {batch_idx}: {e}\033[0m")
            for output in outputs:
                print(f"\033[91mOutput: {output}\033[0m")
            continue
        for sample, parsed_output in zip(batch, parsed_outputs):
            out.append(
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "reasoning_traces": parsed_output,
                    "dataset": sample.dataset,
                    "split": sample.split,
                    "img_path": sample.img_path,
                    "bboxs": sample.bboxs,
                }
            )
        if (batch_idx + 1) % save_every == 0:
            with open(output_json_path, "w") as f:
                json.dump(out, f)
            print(f"Progressive save at batch {batch_idx+1}: {len(out)} samples")

    # Ensure final save
    with open(output_json_path, "w") as f:
        json.dump(out, f)

    print(f"💾 Saved {len(out)} samples to {output_json_path}")
    print("=" * 80)
    print("🎉 Synthetic data creation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    dataset = VCoTData(
        folder_path="/home/gviveiros/.cache/huggingface/hub/datasets--deepcs233--Visual-CoT/snapshots/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata",
        shuffle=False,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda x: x,
    )

    model_id = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
    # model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    if "fp8" in model_id.lower():
        print("Using vllm inference")
        model, processor = load_model_vllm(model_id=model_id)
        inference = partial(batch_inference_vllm, model=model, processor=processor, system_prompt=system_prompt)
    else:
        print("Using transformer inference")
        model, processor = load_model_transformer(model_id=model_id)
        inference = partial(batch_inference_transformer, model=model, processor=processor, system_prompt=system_prompt)

    output_json_path = "/mnt/data-artemis/gviveiros/lantern/LantErn_VisCot_data.json"
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    # create synthetic data
    print("Creating synthetic data...")
    create_synthetic_data(dataloader, inference, output_json_path)

    # print("=" * 40)
    # print("Debugging mode")
    # print("=" * 40)
    # # lets visualize 3 example of each dataset
    # data_mixes = dataset.get_dataset_mix(verbose=False)
    # # iterate over the data_mixes
    # for ds in data_mixes:
    #     print("--------------------------------")
    #     print(f"Dataset: {ds}")
    #     print("--------------------------------")
    #     samples = dataset.copy().filter(lambda x: x["dataset"] == ds)
    #     # shuffle samples
    #     samples = samples.shuffle(seed=42)
    #     for idx, sample in enumerate(samples):
    #         if idx > 1:
    #             break
    #         print("--------------------------------")
    #         output = inference([sample], verbose=False)
    #         print("--------------------------------")
    #         print("Lantern: ", output)
    #         print("--------------------------------")
    #         print("Lantern parsed: ", parse_output(output[0]))
    #         print("--------------------------------")
