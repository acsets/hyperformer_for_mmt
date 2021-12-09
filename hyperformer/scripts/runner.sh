#!/bin/bash

# load some modules here, e.g. cuda

# activate the environment here
python3 -m torch.distributed.launch --nproc_per_node=$nGPU ./finetune_t5_trainer.py $config_file_path