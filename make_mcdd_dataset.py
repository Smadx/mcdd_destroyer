import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义你的数据集路径
dataset_path = 'mcdd_data'
save_path = 'mcdd_dataset'  # 新的保存路径

# 定义转换操作
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 使用ImageFolder加载数据集
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# 创建一个数据加载器
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 迭代数据加载器并保存图像
for idx, (image, label) in tqdm(enumerate(data_loader)):
    # 构建保存路径
    label_folder = os.path.join(save_path, dataset.classes[label])
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    save_image(image, os.path.join(label_folder, f'image_{idx}.jpg'))

print("Dataset saved.")