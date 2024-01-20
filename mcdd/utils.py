import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def make_dataloader(path: str, batch_size: int)-> DataLoader:
    """
    从path中加载数据集,返回与batch_size匹配的DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)