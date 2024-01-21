import torch
from torch import nn

# 搭建一个卷积神经网络,处理图像二分类任务
# 输入图像shape: [1,1,28,28]
class ConvnetMcdd(nn.Module):
    def __init__(self):
        super().__init__()
        # 搭建卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1) # 输出shape: [1,32,28,28]
        self.relu1 = nn.ReLU()
        # 搭建池化层1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出shape: [1,32,14,14]
        # 搭建卷积层2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1) # 输出shape: [1,64,14,14]
        self.relu2 = nn.ReLU()
        # 搭建池化层2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出shape: [1,64,7,7]
        # 搭建全连接层
        self.fc = nn.Linear(in_features=64*7*7, out_features=128) # 输出shape: [1,128]
        self.relu3 = nn.ReLU()
        # 搭建输出层,输出为2分类
        self.out = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        # 卷积层1
        x = self.conv1(x)
        x = self.relu1(x)
        # 池化层1
        x = self.maxpool1(x)
        # 卷积层2
        x = self.conv2(x)
        x = self.relu2(x)
        # 池化层2
        x = self.maxpool2(x)
        # 展平
        x = x.view(-1, 64*7*7)
        # 全连接层
        x = self.fc(x)
        x = self.relu3(x)
        # 输出层
        x = self.out(x)
        return x