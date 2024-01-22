import argparse

from accelerate import Accelerator
import torch
import os
import glob


from utils import (
    init_logger,
    make_single_image,
    log,
)

from mcdd_convnet import ConvnetMcdd
from mcdd_destroyer import mcdd_destroyer

ERROR = -1

def argument():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='data/')
    parser.add_argument("--model-path", type=str, default='model.pt')
    args = parser.parse_args()
    return args


class destroyer_driver:
    def __init__(self, model, cfg, image_shape):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.image_shape = image_shape
        self.model.eval()
        self.load_model()
    
    def load_model(self):
        data = torch.load(self.cfg.model_path)
        log(f"Loading model from {self.cfg.model_path}")
        self.model.load_state_dict(data["model"])

    def predict(self):
        """
        如果data_path下有图片,则预测图片的类别
        """
        if self.cfg.data_path is None or not os.path.isdir(self.cfg.data_path):
            log("No valid directory for prediction")
            return ERROR

        # 查找文件夹中的第一个图片文件
        image_files = glob.glob(os.path.join(self.cfg.data_path, '*'))
        image_files = [f for f in image_files if os.path.isfile(f) and f.lower().endswith(('.jpg'))]

        if len(image_files) == 0:
            log("No valid image file for prediction")
            return ERROR
        
        if len(image_files) > 1:
            log("Warning: more than one image file found, only the first one will be used")

        # 使用第一张图片进行预测
        image_path = image_files[0]
        image = make_single_image(image_path)
        assert image.shape == self.image_shape, "Input image shape is incorrect"
        with torch.no_grad():
            self.model.eval()
            output = self.model.predict(image)
        # 预测完成之后，删除图片
        os.remove(image_path)
        return output
    
if __name__ == "__main__":
    accelerate = Accelerator()
    init_logger(accelerate)
    args = argument()
    model = ConvnetMcdd()
    destroyer = mcdd_destroyer(model, image_shape=(1, 28, 28))
    driver = destroyer_driver(destroyer, args, image_shape=(1, 28, 28))
    flag = True
    while flag:
        result = driver.predict()
        if result == ERROR:
            log(f"Predict result: error")
            flag = False
        else:
            log(f"Predict result: {result}")