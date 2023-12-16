# Antispoofing

[Wandb report link](https://wandb.ai/h1de0us/antispoofing/reports/Antispoofing---Vmlldzo2MTkxMTIx)

## Installation guide

To install all the dependencies

```shell
pip install -r ./requirements.txt
```

If you are trying to start training on kaggle, you may use existing dataset, but you have to copy all the data in the following way:

```shell
!mkdir antispoofing/data
!mkdir antispoofing/data/datasets
!mkdir antispoofing/data/datasets/asvspoof

!cp -r /kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols antispoofing/data/datasets/asvspoof
!cp -r /kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev antispoofing/data/datasets/asvspoof
!cp -r /kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval antispoofing/data/datasets/asvspoof
!cp -r /kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train antispoofing/data/datasets/asvspoof
```

To train a LCNN model, run
```shell
python3 train.py -c src/configs/lcnn.py
```


To train a RawNet2 model, run
```shell
python3 train.py -c src/configs/rawnet.py
```

Test script is provided, but there is no checkpoint yet, so you cannot actually test the trained models :(

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize