import mcdd_convnet
import torch
from torch import nn

# 搭建一个卷积神经网络,处理图像二分类任务
class mcdd_destroyer(nn.Module):
    def __init__(self, model, cfg, image_shape):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.image_shape = image_shape

    def forward(self, batch, label):
        assert batch.shape[1:] == self.image_shape
        

        
        