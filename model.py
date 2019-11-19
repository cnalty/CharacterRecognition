import torch
from torch import nn
import torch.nn.functional as F

class CharNet(nn.Module):
    def __init__(self, out_chars):
        super(CharNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3)



        self.fc1 = nn.Linear(784, out_chars)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool2(out)
        #out = self.conv5(out)
        #out = F.relu(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        return out



