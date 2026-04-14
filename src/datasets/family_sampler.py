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

        # Shuffle within each group, slice into complete chunks, shuffle chunk order
        chunks = []
        for idxs in groups.values():
            idxs = list(idxs)
            rng.shuffle(idxs)
            for start in range(0, len(idxs) - chunk_size + 1, chunk_size):
                chunks.append(idxs[start : start + chunk_size])

        rng.shuffle(chunks)

        # Flatten into ordered index list
        self._ordered_indices = [idx for chunk in chunks for idx in chunk]

    def __len__(self) -> int:
        return len(self._ordered_indices)

    def __getitem__(self, i: int):
        return self.dataset[self._ordered_indices[i]]

    def get_group_key(self, i: int, key: str) -> str:
        return self.dataset.get_group_key(self._ordered_indices[i], key)
