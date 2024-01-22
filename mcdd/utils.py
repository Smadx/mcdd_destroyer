import dataclasses
import warnings

import torch
import torchinfo
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from dataclasses import dataclass
from accelerate import Accelerator
from typing import Optional
from datetime import datetime
from pathlib import Path
from PIL import Image

_accelerator: Optional[Accelerator] = None

@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    seed: int
    results_path: str
    data_path: str
    epochs: int
    model_path: str

def print_model_summary(model, *, batch_size, shape, depth=4, batch_size_torchinfo=1):
    # 打印模型概览
    summary = torchinfo.summary(
        model,
        [(batch_size_torchinfo, *shape)],  # 模型输入尺寸
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # 不显示额外信息
    )

    # 打印概览
    log(summary)

def make_dataloader(path: str, batch_size: int, k: float)-> DataLoader:
    """
    从path中加载数据集,按照比例k划分训练集和验证集,并返回DataLoader
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    n = len(dataset)
    train_size = int(n * k)
    val_size = n - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

def make_single_image(path:str)-> torch.Tensor:
    """
    把path中的图片(只有一张)转换为tensor
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = transform(Image.open(path))
    return image

def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    log(f"Results will be saved to '{results_path}'")
    return results_path

def init_config_from_args(cls, args):
    """
    从args中初始化配置
    """
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})

def init_logger(accelerator: Accelerator):
    global _accelerator
    if _accelerator is not None:
        raise ValueError("Accelerator already set")
    _accelerator = accelerator

def log(message):
    global _accelerator
    if _accelerator is None:
        warnings.warn("Accelerator not set, using print instead.")
        print_fn = print
    else:
        print_fn = _accelerator.print
    print_fn(message)