import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar
from model import Model
import trainer
import math
import numpy

transform_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

testset = torchvision.datasets.ImageFolder(
    root='/home/skye/intel-image/seg_test', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=1)

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

index = 0

for batch_idx, (inputs, targets) in enumerate(testloader):
    for image in inputs:
        image = image.unsqueeze(0)
        target = classes[targets[0]]
        numpy.save(f"{target}/{index}.npy",image.numpy())
        index += 1