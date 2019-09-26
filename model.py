import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
# from utils import progress_bar
import math

criterion = nn.CrossEntropyLoss()


def entropy(tensor):
    c = math.log2(tensor[0].size()[0])
    e_sum = 0
    for i in tensor[0]:
        e_sum += (i*math.log2(i))/c
    return (-e_sum).item()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # print(f"Building basic block {in_planes} => {planes}")
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_layer(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(in_planes, planes, stride))
        in_planes = planes * block.expansion
    return in_planes, nn.Sequential(*layers)


class DeviceMain(nn.Module):
    def __init__(self):
        super(DeviceMain, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.bn1(out))
        return out


class DeviceClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DeviceClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.linear = nn.Linear(3072, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.softmax(out)
        return out


class DeviceExtract(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(DeviceExtract, self).__init__()
        self.in_planes, self.layer1 = make_layer(
            block, 64, 48, num_blocks[0], stride=1)
        self.pool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool(out)
        return out


class EdgeMain(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(EdgeMain, self).__init__()
        self.in_planes, self.layer2 = make_layer(
            block, 48, 128, num_blocks[1], stride=2)

    def forward(self, x):
        return self.layer2(x)


class EdgeClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EdgeClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.linear = nn.Linear(3200, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.softmax(out)
        return out


class Cloud(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=6):
        super(Cloud, self).__init__()
        self.in_planes, self.layer3 = make_layer(
            block, 128, 256, num_blocks[2], stride=2)
        _, self.layer4 = make_layer(
            block, self.in_planes, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.softmax(out)
        return out


class Model:
    def __init__(self,jit=False):
        if jit:
            self.device_main = torch.jit.load('device_main.pt')
            self.device_extract = torch.jit.load('device_extract.pt')
            self.device_classifier = torch.jit.load('device_classifier.pt')
            self.edge_classifier = torch.jit.load('edge_classifier.pt')
            self.edge_main = torch.jit.load('edge_main.pt')
            self.cloud = torch.jit.load('cloud.pt')
        else:
            self.device_main = DeviceMain()
            self.device_classifier = DeviceClassifier()
            self.device_extract = DeviceExtract()

            self.edge_main = EdgeMain()
            self.edge_classifier = EdgeClassifier()
            self.cloud = Cloud()

        self.main_model = MainRoadModel(
            self.device_main, self.device_extract, self.edge_main, self.cloud)
        self.device_classifier_model = DeviceClassifierModel(
            self.device_main, self.device_classifier, self.device_extract)
        self.edge_classifier_model = EdgeClassifierModel(
            self.device_main, self.device_extract,
            self.edge_main, self.edge_classifier)

        if not os.path.isdir('split_ckp'):
            os.mkdir('split_ckp')

    def cuda(self):
        self.main_model.cuda()
        self.device_classifier.cuda()
        self.edge_classifier.cuda()

    def cpu(self):
        self.main_model.cpu()
        self.device_classifier.cpu()
        self.edge_classifier.cpu()

    def save_main(self):
        print("Saving Main Model...")
        torch.save(self.device_main.state_dict(),
                   './split_ckp/device_main.pth')
        torch.save(self.device_extract.state_dict(),
                   './split_ckp/device_extract.pth')
        torch.save(self.edge_main.state_dict(), './split_ckp/edge_main.pth')
        torch.save(self.cloud.state_dict(), './split_ckp/cloud.pth')

    def save_device(self):
        print("Saving Device Classifier...")
        torch.save(self.device_classifier.state_dict(),
                   './split_ckp/device_classifier.pth')

    def save_edge(self):
        print("Saving Edge Classifier...")
        torch.save(self.edge_classifier.state_dict(),
                   './split_ckp/edge_classifier.pth')

    def load_device(self,device,path='./split_ckp/'):
        self.device_classifier.load_state_dict(
            torch.load(path+'device_classifier.pth',map_location=torch.device(device)))

    def load_edge(self,device):
        self.edge_main.load_state_dict(torch.load('./split_ckp/edge_main.pth',map_location=device))
        self.edge_classifier.load_state_dict(
            torch.load('./split_ckp/edge_classifier.pth',map_location=torch.device(device)))

    def load_main(self,device):
        self.device_main.load_state_dict(
            torch.load('./split_ckp/device_main.pth',map_location=torch.device(device)))
        self.device_extract.load_state_dict(
            torch.load('./split_ckp/device_extract.pth',map_location=torch.device(device)))
        self.edge_main.load_state_dict(torch.load('./split_ckp/edge_main.pth',map_location=torch.device(device)))
        self.cloud.load_state_dict(torch.load('./split_ckp/cloud.pth',map_location=torch.device(device)))

    def load(self,device,path='.'):
        self.device_main.load_state_dict(
            torch.load(path+'/split_ckp/device_main.pth',map_location=torch.device(device)))
        self.device_extract.load_state_dict(
            torch.load(path+'/split_ckp/device_extract.pth',map_location=torch.device(device)))
        self.device_classifier.load_state_dict(
            torch.load(path+'/split_ckp/device_classifier.pth',map_location=torch.device(device)))
        self.edge_classifier.load_state_dict(
            torch.load(path+'/split_ckp/edge_classifier.pth',map_location=torch.device(device)))
        self.edge_main.load_state_dict(torch.load(path+'/split_ckp/edge_main.pth',map_location=torch.device(device)))
        self.cloud.load_state_dict(torch.load(path+'/split_ckp/cloud.pth',map_location=torch.device(device)))

    def save_jit(self):
        self.device_main.eval()
        self.device_classifier.eval()
        self.device_extract.eval()
        self.edge_classifier.eval()
        self.edge_main.eval()
        self.cloud.eval()
        torch.jit.trace(self.device_main, torch.rand(1, 3, 150, 150)).save("device_main.pt")
        torch.jit.trace(self.device_classifier, torch.rand(1, 48, 5, 5)).save("device_classifier.pt")
        torch.jit.trace(self.device_extract, torch.rand(1, 64, 150, 150)).save("device_extract.pt")
        torch.jit.trace(self.edge_main, torch.rand(1, 48, 5, 5)).save("edge_main.pt")
        torch.jit.trace(self.edge_classifier, torch.rand(1, 128, 5, 5)).save("edge_classifier.pt")
        torch.jit.trace(self.cloud, torch.rand(1, 128, 3, 3)).save("cloud.pt")


    def test(self, testloader, device):
        self.device_main.eval()
        self.device_classifier.eval()
        self.device_extract.eval()
        self.edge_classifier.eval()
        self.edge_main.eval()
        self.cloud.eval()

        device_test_loss = 0
        device_test_loss_total = 0
        edge_test_loss = 0
        edge_test_loss_total = 0
        cloud_test_loss = 0
        cloud_test_loss_total = 0

        device_correct = 0
        edge_correct = 0
        cloud_correct = 0

        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                device_main_outputs = self.device_main(inputs)
                device_to_edge = self.device_extract(device_main_outputs)
                device_outputs = self.device_classifier(device_to_edge)

                edge_main_outputs = self.edge_main(device_to_edge)
                edge_outputs = self.edge_classifier(edge_main_outputs)

                cloud_outputs = self.cloud(edge_main_outputs)

                device_loss = criterion(device_outputs, targets)
                edge_loss = criterion(edge_outputs, targets)
                cloud_loss = criterion(cloud_outputs, targets)

                device_test_loss_total += device_loss.item()
                device_test_loss = device_test_loss_total/(batch_idx+1)
                edge_test_loss_total += edge_loss.item()
                edge_test_loss = edge_test_loss_total/(batch_idx+1)
                cloud_test_loss_total += cloud_loss.item()
                cloud_test_loss = cloud_test_loss_total/(batch_idx+1)

                device_outputs = nn.Softmax(-1)(device_outputs)
                edge_outputs = nn.Softmax(-1)(edge_outputs)
                cloud_outputs = nn.Softmax(-1)(cloud_outputs)

                device_prob, device_predicted = device_outputs.max(1)
                edge_prob, edge_predicted = edge_outputs.max(1)
                cloud_prob, cloud_predicted = cloud_outputs.max(1)

                if batch_idx % 100 == 0:
                    print("device", device_prob)
                    print("edge", edge_prob)
                    print("cloud", cloud_prob)

                total += targets.size(0)
                device_correct += device_predicted.eq(targets).sum().item()
                edge_correct += edge_predicted.eq(targets).sum().item()
                cloud_correct += cloud_predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f,%.3f,%.3f | Acc: %.3f%%,%.3f%%,%.3f%%'
                             % (device_test_loss, edge_test_loss, cloud_test_loss, 100.*device_correct/total, 100.*edge_correct/total, 100.*cloud_correct/total))

    def test_overall(self, testloader, device, early_t, mid_t):
        self.device_main.eval()
        self.device_classifier.eval()
        self.device_extract.eval()
        self.edge_classifier.eval()
        self.edge_main.eval()
        self.cloud.eval()

        correct = 0
        early_exit = 0
        mid_exit = 0

        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                total += targets.size(0)
                print(targets)
                inputs, targets = inputs.to(device), targets.to(device)
                device_main_outputs = self.device_main(inputs)
                device_to_edge = self.device_extract(device_main_outputs)
                device_outputs = self.device_classifier(device_to_edge)

                device_outputs = nn.Softmax(-1)(device_outputs)
                device_t = entropy(device_outputs)
                if device_t < early_t:
                    _, device_predicted = device_outputs.max(1)
                    early_exit += 1
                    mid_exit += 1
                    correct += device_predicted.eq(targets).sum().item()
                else:
                    edge_main_outputs = self.edge_main(device_to_edge)
                    edge_outputs = self.edge_classifier(edge_main_outputs)
                    edge_outputs = nn.Softmax(-1)(edge_outputs)
                    edge_t = entropy(edge_outputs)
                    if edge_t < mid_t:
                        _, edge_predicted = edge_outputs.max(1)
                        correct += edge_predicted.eq(targets).sum().item()
                        mid_exit += 1
                    else:
                        cloud_outputs = self.cloud(edge_main_outputs)
                        _, cloud_predicted = cloud_outputs.max(1)
                        correct += cloud_predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Overall Acc: %.3f%% (%d/%d) | Early Exit Rate: %.3f%% | Mid Exit Rate: %.3f%%'
                             % (100.*correct/total, correct, total, 100.*early_exit/total, 100.*mid_exit/total))


class MainRoadModel(nn.Module):
    def __init__(self, device_main: DeviceMain, device_extract: DeviceExtract, edge_main: EdgeMain, cloud: Cloud):
        super(MainRoadModel, self).__init__()
        self.device_main = device_main
        self.device_extract = device_extract
        self.edge_main = edge_main
        self.cloud = cloud

    def forward(self, x):
        out = self.device_main(x)
        out = self.device_extract(out)
        out = self.edge_main(out)
        out = self.cloud(out)
        return out


class DeviceClassifierModel(nn.Module):
    def __init__(self, device_main: DeviceMain, device_classifier: DeviceClassifier, device_extract: DeviceExtract):
        super(DeviceClassifierModel, self).__init__()
        self.device_main = device_main
        self.device_extract = device_extract
        self.device_classifier = device_classifier

    def forward(self, x):
        out = self.device_main(x)
        out = self.device_extract(out)
        return self.device_classifier(out)


class EdgeClassifierModel(nn.Module):
    def __init__(self, device_main: DeviceMain, device_extract: DeviceExtract, edge_main: EdgeMain, edge_classifier: EdgeClassifier):
        super(EdgeClassifierModel, self).__init__()
        self.device_main = device_main
        self.device_extract = device_extract
        self.edge_main = edge_main
        self.edge_classifier = edge_classifier

    def forward(self, x):
        out = self.device_main(x)
        out = self.device_extract(out)
        out = self.edge_main(out)
        return self.edge_classifier(out)
