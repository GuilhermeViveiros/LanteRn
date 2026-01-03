"""
Dataset for supervised fine-tuning (SFT)
"""
import os
from functools import partial
from typing import Optional
from torch.utils.data import Dataset
from datasets import load_dataset
from src.rl.utils import convert_example
import logging

logger = logging.getLogger("LantErn-GRPO-Dataset")

class GRPODataset(Dataset):
    def __init__(self, data_path: str, image_root: str, system_prompt: Optional[str] = None):
        # Load data
        ds = load_dataset(
            "json",
            data_files=data_path,
            split="train"
        )

        # Create a stable 'images' list column containing absolute paths
        def add_images(ex, image_root: str):
            ex["images"] = [os.path.join(image_root, ex["image"])]
            return ex

        ds = ds.map(partial(add_images, image_root=image_root))

        # Convert to GRPO dataset columns (and drop old columns)
        ds = ds.map(
            lambda ex: convert_example(ex, system_prompt), remove_columns=ds.column_names
        )
        
        # add the dataset
        self.ds = ds
        # log the number of examples
        logger.info(f"\033[92mNumber of examples in GRPODataset: {len(self.ds)}\033[0m")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
