"""
FamilyGroupedDataset
--------------------
Map-style Dataset that pre-computes family-grouped indices so that consecutive
chunks of `chunk_size` samples all share the same `group_key` value
(e.g. "shape_C_name"). This ensures each batch contains hard negatives —
samples with the same query shape but different required transformations —
which improves the InfoNCE contrastive loss signal.

Must be used with SequentialSampler (or no sampler) to preserve family order.
LantErnSFTrainer overrides _get_train_sampler to use SequentialSampler when
the train_dataset is a FamilyGroupedDataset instance.

Usage:
    dataset = FamilyGroupedDataset(
        dataset=train_ds,
        chunk_size=per_device_batch * num_gpus,
        group_key="shape_C_name",
        seed=42,
    )
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from torch.utils.data import Dataset


class FamilyGroupedDataset(Dataset):
    """
    Map-style wrapper that pre-orders indices so consecutive chunks of
    `chunk_size` items share the same `group_key` value.  Must be paired
    with SequentialSampler so the Trainer respects the ordering.
    """

    def __init__(
        self,
        dataset,
        chunk_size: int,
        group_key: str = "shape_C_name",
        seed: int = 42,
    ):
        super().__init__()
        self.dataset    = dataset
        self.chunk_size = chunk_size
        self.group_key  = group_key
        self.seed       = seed

        rng = random.Random(seed)

        # Build {group_value: [indices]} — cheap, no image I/O
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            key = dataset.get_group_key(i, group_key)
            groups[key].append(i)

        # 'groups' now maps each unique group_key value (object) to its sample indices
        # keys = different objects, values = lists of sample indices for each object
        total_samples = len(dataset)
        for obj_name, sample_idxs in groups.items():
            percent = (len(sample_idxs) / total_samples) * 100 if total_samples > 0 else 0.0
            logging.info(f"[FamilyGroupedDataset] object '{obj_name}': {len(sample_idxs)} samples ({percent:.2f}%)")

        # Build chunks where every sample has a unique intermediate_key
        # (i.e. unique rotation strip — same shape_C but different rot_C_idx or rot_step).
        # Strategy: group by intermediate_key, then round-robin across keys to fill chunks.
        chunks = []
        for idxs in groups.values():
            # Sub-group by intermediate_key
            inter_groups: dict[str, list[int]] = defaultdict(list)
            for idx in idxs:
                ikey = dataset.get_group_key(idx, "intermediate_key")
                inter_groups[ikey].append(idx)
            # Shuffle within each sub-group
            for sub in inter_groups.values():
                rng.shuffle(sub)
            # Round-robin across unique intermediate_keys to build chunks
            # Each chunk picks one sample from each distinct key in turn.
            keys = list(inter_groups.keys())
            rng.shuffle(keys)
            pointers = {k: 0 for k in keys}
            while True:
                chunk = []
                used_keys = []
                for k in keys:
                    if pointers[k] < len(inter_groups[k]):
                        chunk.append(inter_groups[k][pointers[k]])
                        pointers[k] += 1
                        used_keys.append(k)
                    if len(chunk) == chunk_size:
                        break
                if len(chunk) < chunk_size:
                    if chunk:
                        chunks.append(chunk)
                    break  # not enough unique keys left for a full chunk
                chunks.append(chunk)
                rng.shuffle(keys)  # vary key order across chunks

        rng.shuffle(chunks)

        # Flatten into ordered index list
        self._ordered_indices = [idx for chunk in chunks for idx in chunk]

    def __len__(self) -> int:
        return len(self._ordered_indices)

    def __getitem__(self, i: int):
        return self.dataset[self._ordered_indices[i]]

    def get_group_key(self, i: int, key: str) -> str:
        return self.dataset.get_group_key(self._ordered_indices[i], key)
