# DIWM
Code for the paper entitled "Data-Independent Black-Box Watermark and Unforgeable Ownership Proof for Deep Neural Networks"

## Preparation
Incorporate baseline models them directly into /checkpoint/teacher.

To generate anchors, refer to the instruction in https://github.com/VainF/Data-Free-Adversarial-Distillation. 

Notice that you should adopt a different auxiliary backbone. 

Results are saved in /checkpoint/student .

## Running DIWM
Execute KFWM_XX for the respective task. 

## Requirements:
PyTorch>=1.11

CUDA
