import torch

import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # items are already padded inside the ASVSpoofDataset class
    assert all([item[0].shape == dataset_items[0][0].shape for item in dataset_items]) # (batch_size, n_channels, 64000)
    return {
        "audios": torch.stack([item[0] for item in dataset_items]),
        "labels": torch.as_tensor([item[1] for item in dataset_items]),
    }