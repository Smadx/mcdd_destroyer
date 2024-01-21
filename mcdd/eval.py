import argparse
from pathlib import Path

import torch
import yaml
from accelerate.utils import set_seed
from tqdm import tqdm

from utils import (
    TrainConfig,
    make_dataloader,
    init_logger,
    init_config_from_args,
    log,
    print_model_summary,
    get_date_str,
)

from mcdd_convnet import ConvnetMcdd
from mcdd_destroyer import mcdd_destroyer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Load config from YAML.
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    model = ConvnetMcdd()
    print_model_summary(model, batch_size=cfg.batch_size, shape=(1, 28, 28))
    train_loader, val_loader = make_dataloader(cfg.data_path, cfg.batch_size, 0.8)
    destroyer = mcdd_destroyer(model, cfg, image_shape=(1, 28, 28))
    Evaluator(
        destroyer,
        train_loader,
        val_loader,
        config=cfg,
        eval_batch_size=args.batch_size,
        results_path=Path(args.results_path),
    ).eval()

class Evaluator:
    def __init__(self, model, train_loader, val_loader, config, results_path, eval_batch_size):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.path = results_path

        self.model = model.eval()
        self.eval_batch_size = eval_batch_size

        self.eval_path = self.path / f"eval_{get_date_str()}"
        self.eval_path.mkdir()
        self.checkpoint_path = self.path / "model.pt"

    def load_checkpoint(self):
        data = torch.load(self.checkpoint_path)
        log(f"Loading checkpoint from {self.checkpoint_path}")
        self.model.load_state_dict(data["model"])

    def eval(self):
        log("Evaluating model")
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                for i in range(images.shape[0]):  # 遍历批次中的每张图片
                    image = images[i].unsqueeze(0)  # 获取单张图片并增加批次维度
                    label = labels[i]
                    predicted_class = self.model.predict(image)  # 使用 predict 函数进行预测
                    total += 1
                    correct += (predicted_class == label).item()

        accuracy = 100 * correct / total
        log(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()