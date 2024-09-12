# TARL
This repository contains code corresponding to "TARL: A Highly Efficient Black-Box Adversarial Attack with Reinforcement Learning" 

# Repo Overview 
All attack pipelines were created by modifying the [official Triangle Attack repository](https://github.com/xiaosen-wang/TA). Code from this repository is also provided here to run the original Triangle Attack. There are two main attack pipelines featured in this repository:

- **TA_Batches.py**: Takes the 200 images provided by the Triangle Attack repository (needs to be inserted here!) and runs either TA or TARL.
- **TA_ImageNet_Batches.py**: Given a folder containing ImageNet examples and a text file of their labels, run either TA or TARL on each example in sequence.

Both TA_Batches and TA_ImageNet are completely automated to load models with pre-processing wrappers, attack these models, then save the adversarial results. During the attack phase, examples are processed in sequence to be as GPU-friendly and resource unintensive as possible. 

Each attack pipeline has two global variables:
- **model**: Specify model to run attack on. Model is generated from attack_utils, which loads weights and uses wrappers to amend pre-processing issues when running attacks; note that weights for ViT-L, ViT-B and BiT-M models have to be placed in this repository!
- **use_RL**: Boolean that is set to True to run TARL, set to False to run TA. 

# Requirements
.yml with necessary libraries are provided. 

# System Overview
Experiments and validation were conducted on a NVIDIA Quadro RTX 8000. 
