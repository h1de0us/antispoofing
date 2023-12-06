from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import src.datasets
from src.collate_fn.collate import collate_fn
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, src.datasets, config_parser=configs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert "batch_size" in params, \
            "You must provide batch_size for each split"
    
        bs = params["batch_size"]
        shuffle = True

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
        )
        dataloaders[split] = dataloader
    return dataloaders
