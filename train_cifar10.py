# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

torch.manual_seed(0)
np.random.seed(0)

class RandomResize(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        size = torch.randint(self.size-10, self.size+10, (1,))      
        return torchvision.transforms.functional.resize(img, size.item())


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--grad_acc_steps', default='1', type=int, help="number of gradient accumulation steps before backward pass")

args = parser.parse_args()

bs = int(args.bs)
imsize = int(args.size)

use_amp = bool(~args.noamp)
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

'''transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 12,
    heads = 12,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    from timm.models import create_model
    net = create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
normalize_fn = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
with open(f'log/log_{args.net}_patch{args.patch}.txt', 'w') as appender:
    appender.write("\n")

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    step = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        step += 1
        inputs, targets = inputs.to(device), targets.to(device)

        batch_img_size = np.random.choice([384, 368, 352, 336, 320, 304, 288, 272, 256, 240, 224, 208, 192, 176, 160, 144, 128, 112], 1)
        resize_fn = transforms.Resize(int(batch_img_size[0]))
        inputs = resize_fn(inputs)
        inputs = normalize_fn(inputs)

        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()

        if step % args.grad_acc_steps == 0 or (step == (len(trainloader) - 1)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total))
    return train_loss/(batch_idx+1)
    

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_image_sizes = [224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384]
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            size_index = 0
            total += targets.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            while size_index < len(test_image_sizes):
                resize_fn = transforms.Resize(test_image_sizes[size_index])
                resized_inputs = resize_fn(inputs.clone())
                resized_inputs = normalize_fn(resized_inputs)
                outputs = net(resized_inputs)
                confidence, predicted = F.softmax(outputs, dim=1).max(1)
                if confidence >= 0.9:
                    correct += predicted.eq(targets).sum().item()
                    break
                size_index += 1

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% '% (100.*correct/total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
    
        os.makedirs("log", exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, acc: {(acc):.5f}, best_acc: {(best_acc):.5f}'
        print(content)
        with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
                appender.write(content + "\n")
            
    return acc

list_acc = []
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    list_acc.append(acc)

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_acc) 




    
