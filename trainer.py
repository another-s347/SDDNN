import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import progress_bar
from model import Model

criterion = nn.CrossEntropyLoss()


class MainTrainer:
    def __init__(self, model: Model, writer, trainloader, testloader, device):
        self.model = model
        self.writer = writer
        self.trainloader = trainloader
        self.testloader = testloader
        self.best_acc = 0
        self.best_epoch = 0
        self.device = device

    def train(self, epoch, lr):
        self.model.device_main.train()
        self.model.device_extract.train()
        self.model.edge_main.train()
        self.model.cloud.train()
        optimizer = optim.SGD(self.model.main_model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=5e-4)
        print(f'\nTrain:Main, Epoch:{epoch}')
        train_loss = 0
        train_loss_total = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model.device_main(inputs)
            outputs = self.model.device_extract(outputs)
            outputs = self.model.edge_main(outputs)
            outputs = self.model.cloud(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss = train_loss_total/(batch_idx+1)
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), '[Main] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss, 100.*correct/total, correct, total))

        acc = correct/total
        loss = train_loss
        self.writer.add_scalar('[Main] Loss/train', loss, epoch)
        self.writer.add_scalar('[Main] Accuracy/train', 100.*acc, epoch)

    def test(self, epoch):
        self.model.device_main.eval()
        self.model.device_extract.eval()
        self.model.edge_main.eval()
        self.model.cloud.eval()
        acc, loss = test("Main", epoch, self.model.main_model,
                         self.device, self.testloader)
        if acc > self.best_acc:
            self.writer.add_scalar('[Main] Loss/test', loss, epoch)
            self.writer.add_scalar('[Main] Accuracy/test',
                                   100.*acc, epoch)
            self.best_acc = acc
            self.best_epoch = epoch
            self.model.save_main()
            return True
        else:
            return False

    def load(self):
        self.model.load_main()
        return self.best_epoch


class DeviceTrainer:
    def __init__(self, model: Model, writer, trainloader, testloader, device):
        self.model = model
        self.writer = writer
        self.trainloader = trainloader
        self.testloader = testloader
        self.best_acc = 0
        self.best_epoch = 0
        self.device = device

    def train(self, epoch, lr):
        print(f'\nTrain:Device, Epoch:{epoch}')
        self.model.device_main.eval()
        self.model.device_classifier.train()
        optimizer = optim.SGD(self.model.device_classifier.parameters(
        ), lr=lr, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss_total = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model.device_main(inputs)
            outputs = self.model.device_classifier(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss = train_loss_total/(batch_idx+1)
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), '[Device] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss, 100.*correct/total, correct, total))

        acc = correct/total
        loss = train_loss
        self.writer.add_scalar('[Device] Loss/train', loss, epoch)
        self.writer.add_scalar('[Device] Accuracy/train', 100.*acc, epoch)

    def test(self, epoch):
        self.model.device_main.eval()
        self.model.device_classifier.eval()
        acc, loss = test(
            "Device", epoch, self.model.device_classifier_model, self.device, self.testloader)
        if acc > self.best_acc:
            self.writer.add_scalar('[Device] Loss/test', loss, epoch)
            self.writer.add_scalar('[Device] Accuracy/test',
                                   100.*acc, epoch)
            self.best_acc = acc
            self.best_epoch = epoch
            self.model.save_device()
            return True
        else:
            return False

    def load(self):
        self.model.load_device()
        return self.best_epoch


class EdgeTrainer:
    def __init__(self, model: Model, writer, trainloader, testloader, device):
        self.model = model
        self.writer = writer
        self.trainloader = trainloader
        self.testloader = testloader
        self.best_acc = 0
        self.best_epoch = 0
        self.device = device

    def train(self, epoch, lr):
        print(f'\nTrain:Edge, Epoch:{epoch}')
        self.model.device_main.eval()
        self.model.device_extract.eval()
        self.model.edge_classifier.train()
        optimizer = optim.SGD(self.model.edge_classifier.parameters(
        ), lr=lr, momentum=0.9, weight_decay=5e-4)
        train_loss_total = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model.device_main(inputs)
            outputs = self.model.device_extract(outputs)
            outputs = self.model.edge_classifier(outputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss = train_loss_total/(batch_idx+1)
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), '[Edge] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss, 100.*correct/total, correct, total))

        acc = correct/total
        loss = train_loss
        self.writer.add_scalar('[Edge] Loss/train', loss, epoch)
        self.writer.add_scalar('[Edge] Accuracy/train', 100.*acc, epoch)

    def test(self, epoch):
        self.model.device_main.eval()
        self.model.device_extract.eval()
        self.model.edge_classifier.eval()
        acc, loss = test(
            "Edge", epoch, self.model.edge_classifier_model, self.device, self.testloader)
        if acc > self.best_acc:
            self.writer.add_scalar('[Edge] Loss/test', loss, epoch)
            self.writer.add_scalar('[Edge] Accuracy/test',
                                   100.*acc, epoch)
            self.best_acc = acc
            self.best_epoch = epoch
            self.model.save_edge()
            return True
        else:
            return False

    def load(self):
        self.model.load_edge()
        return self.best_epoch


def test(name, epoch, model, device, testloader):
    test_loss = 0
    test_loss_total = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss_total += loss.item()
            test_loss = test_loss_total/(batch_idx+1)
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), '[%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (name, test_loss, 100.*correct/total, correct, total))

    acc = correct/total
    return acc, test_loss