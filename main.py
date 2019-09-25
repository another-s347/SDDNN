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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.ImageFolder(
    root='/home/skye/intel-image/seg_train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    root='/home/skye/intel-image/seg_test', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=1)

writer = SummaryWriter()

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def save_jit():
    model = Model()
    model.load_main(device)
    model.load_device(device)
    model.load_edge(device)
    model.save_jit()

def test_jit():
    model = Model(jit=True)
    model.test_overall(testloader, device, 0.2, 0.1)


def test():
    model = Model()
    if device == 'cuda':
        model.cuda()
        cudnn.benchmark = True
    model.load_main(device)
    model.load_device(device)
    model.load_edge(device)
    # for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     print(f"Current t: {t}")
    model.test_overall(testloader, device, 0.2, 0.1)


def train():
    model = Model()
    if device == 'cuda':
        model.cuda()
        cudnn.benchmark = True

    main_trainer = trainer.MainTrainer(
        model, writer, trainloader, testloader, device)
    train_loop(main_trainer, 0.1)
    model.load_main()
    device_trainer = trainer.DeviceTrainer(
        model, writer, trainloader, testloader, device)
    train_loop(device_trainer, 0.1)
    edge_trainer = trainer.EdgeTrainer(
        model, writer, trainloader, testloader, device)
    train_loop(edge_trainer, 0.1)


def pred(name, model, predloader):
    model.eval()
    for batch_idx, (inputs, unused_targets) in predloader:
        if batch_idx % 2000 == 0:
            for image in inputs:
                image = image.unsqueeze(0).to(device)
                early_outputs, outputs = model(image)

                prob, predicted = outputs.max(1)
                early_prob, early_predicted = early_outputs.max(1)
                tag = f"Task:{name}, Device:{early_prob}/{classes[early_predicted]}, Edge:{prob}/{classes[predicted]}"
                writer.add_image(tag, image)


def train_loop(trainer, lr):
    print(f"Current learning rate: {lr}")
    for epoch in range(0, 80):
        trainer.train(epoch, lr)
        if trainer.test(epoch):
            no_progress = 0
        else:
            no_progress += 1
        if no_progress >= 6:
            break
    lr = lr/10
    for i in range(0, 2):
        print(f"Current learning rate: {lr}")
        start_epoch = trainer.load()
        no_progress = 0
        for epoch in range(start_epoch, start_epoch+80):
            trainer.train(epoch, lr)
            if trainer.test(epoch):
                no_progress = 0
            else:
                no_progress += 1
            if no_progress >= 6:
                break
        lr = lr/10


if __name__ == '__main__':
    test()
