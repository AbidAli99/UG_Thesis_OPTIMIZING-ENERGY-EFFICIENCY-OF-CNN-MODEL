import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelComp4(nn.Module):
    def __init__(self):
        super(ModelComp2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False, padding=2)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False, padding=2)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False, padding=2)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(160 * 3 * 3, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        pool1 = self.pool1(conv2)
        conv3 = F.relu(self.conv3_bn(self.conv3(pool1)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        pool2 = self.pool2(conv4)
        conv5 = F.relu(self.conv5_bn(self.conv5(pool2)))
        pool3 = self.pool3(conv5)
        flat1 = torch.flatten(pool3, 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)


