# Antispoofing

## Installation guide

< Write your installation guide here >

```shell
pip install -r ./requirements.txt
```



## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize