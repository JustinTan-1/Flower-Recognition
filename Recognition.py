import torch.nn as nn
import torch.nn.functional as F

class Recognition2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 102)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1((self.conv1(x)))))
        x = self.pool(F.relu(self.bn2((self.conv2(x)))))
        x = self.pool(F.relu(self.bn3((self.conv3(x)))))
        x = self.pool(F.relu(self.bn4((self.conv4(x)))))
        x = self.pool(F.relu(self.bn5((self.conv5(x)))))
        x = self.pool(F.relu(self.bn6((self.conv6(x)))))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1(x)
        return x