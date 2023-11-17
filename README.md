# Input Compression with Positional Consistency for Efficient Training and Inference of Transformer Neural Networks


This repository contains code for the paper Input Compression with Positional Consistency for Efficient Training and Inference of Transformer Neural Networks.

## Dependencies
To install the necessary packages, use `pip install -r requirements.txt `

This code was tested on Tested on RTX 2080 Ti GPU with CUDA Version 11.4

## Commands
+ To fine-tune the ViT-Base-384/16 model on CIFAR-10, use `python train_cifar10.py --net vit_timm --bs 8 --grad_acc_steps 64 --lr 0.5e-5 `.
+ To fine-tune the ViT-Base-384/16 model on CIFAR-100, use `python train_cifar100.py --net vit_timm --bs 8 --grad_acc_steps 64 --lr 0.5e-5 `.

Both commands use ICPC for both training and inference.
