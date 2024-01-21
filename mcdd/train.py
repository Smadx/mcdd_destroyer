import argparse
from argparse import BooleanOptionalAction

import torch
import yaml
import os
import dataclasses
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path

from tqdm import tqdm
from utils import (
    TrainConfig,
    make_dataloader,
    init_logger,
    init_config_from_args,
    log,
    print_model_summary,
    handle_results_path,
)
from mcdd_convnet import ConvnetMcdd
from mcdd_destroyer import mcdd_destroyer

def main():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    cfg = init_config_from_args(TrainConfig, args)

    model = ConvnetMcdd()
    print_model_summary(model, batch_size=cfg.batch_size, shape=(1, 28, 28))
    with accelerator.local_main_process_first():
        train_loader, val_loader = make_dataloader(cfg.data_path, cfg.batch_size, 0.8)
    destroyer = mcdd_destroyer(model, cfg, image_shape=(1, 28, 28))
    Trainer(destroyer,
            train_loader,
            val_loader,
            accelerator,
            make_opt=lambda params: torch.optim.Adam(params, cfg.lr, eps=1e-8), 
            config=cfg, 
            results_path=handle_results_path(args.results_path)
        ).train()

class Trainer:
    def __init__(self, model, train_loader, val_loader, accelerator, make_opt, config, results_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.cfg = config
        self.results_path = results_path
        self.checkpoint_file = self.results_path / f"model.pt"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(config), f)

    def train(self):
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            for batch, label in tqdm(self.train_loader):
                self.opt.zero_grad()
                loss = self.model(batch, label)
                self.accelerator.backward(loss)
                self.opt.step()
                running_loss += loss.item()
            print(f"Epoch {epoch} loss: {running_loss / len(self.train_loader)}")
        self.save()
    
    def save(self):
        """
        把模型保存到指定路径
        """
        self.model.eval()
        checkpoint_path = Path(self.checkpoint_file)
        checkpoint_dir = checkpoint_path.parent

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.model.state_dict(), checkpoint_path)
        log(f"Saved model to {checkpoint_path}")

if __name__ == "__main__":
    main()