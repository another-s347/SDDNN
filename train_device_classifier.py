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
from model import Model
import math
import numpy
import sys

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model()
model.load(device)
if os.path.isfile('/tmp/sddnn/device_classifier.pth'):
    model.load_device(device,'/tmp/sddnn/')

def test():
    model.device_classifier.eval()
    total = 0
    correct = 0
    for classes in [0,1,2,3,4,5]:
        file_list = os.listdir(f"/tmp/sddnn/test_set/{classes}/")
        buildings = [numpy.load(f"/tmp/sddnn/test_set/{classes}/"+x) for x in file_list]
        t = [torch.from_numpy(n) for n in buildings]
        for inputs in t:
            targets = torch.tensor([classes])
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.device_classifier(inputs)
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct/total

def train(old_acc):
    model.device_classifier.train()
    total = 0
    correct = 0
    optimizer = optim.SGD(model.device_classifier.parameters(), lr=0.000001, momentum=0.9, weight_decay=5e-5)
    for classes in [0,1,2,3,4,5]:
        file_list = os.listdir(f"/tmp/sddnn/train_set/{classes}/")
        buildings = [numpy.load(f"/tmp/sddnn/train_set/{classes}/"+x) for x in file_list]
        t = [torch.from_numpy(n) for n in buildings]
        for inputs in t:
            targets = torch.tensor([classes])
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model.device_classifier(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    new_acc = test()
    if new_acc > old_acc:
        print("trained")
        torch.save(model.device_classifier.state_dict(),'/tmp/sddnn/device_classifier.pth')
        torch.jit.trace(model.device_classifier, torch.rand(1, 48, 5, 5)).save("/tmp/sddnn/device_classifier.pt")
        sys.exit(0)
    else:
        print(f"not trained, old = {old_acc}, new = {new_acc}")
        sys.exit(1)
    
train(test())