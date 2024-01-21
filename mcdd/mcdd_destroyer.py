import torch
from torch import nn

# 搭建一个卷积神经网络,处理图像二分类任务
class mcdd_destroyer(nn.Module):
    def __init__(self, model, cfg, image_shape):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.image_shape = image_shape
        self.loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

    def forward(self, batch, label):
        assert batch.shape[1:] == self.image_shape
        # 使用ConvnetMcdd模型处理批次数据
        output = self.model(batch)

        # 计算交叉熵损失
        loss = self.loss_fn(output, label)

        return loss

    def predict(self, image):
        # 确保模型处于评估模式
        self.eval()

        # 检查输入图片的形状，如果需要，添加一个维度
        if image.shape == (1, 28, 28):
            image = image.unsqueeze(0)  # 增加一个维度，使形状变为 [1, 1, 28, 28]

        # 确保输入图片的形状正确
        assert image.shape == (1, 1, 28, 28), "Input image shape is incorrect"

        # 处理单张图片
        with torch.no_grad():
            output = self.model(image)  # 使用内部模型获取输出

            # 应用Softmax来获取概率分布
            probabilities = torch.softmax(output, dim=1)

            # 获取最可能的类别
            _, predicted_class = torch.max(probabilities, 1)

            return predicted_class.item()
        