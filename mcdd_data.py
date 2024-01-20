from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
import random
from tqdm import tqdm

# 图像文件夹路径
image_folder = 'negative_samples_jpg'

# 数据增强后图像的存储路径
augmented_folder = 'augmented_negative_samples_jpg'
if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

# 定义数据增强函数
def augment_image(image):
    # 执行随机旋转
    image = image.rotate(random.randint(-30, 30))
    
    # 添加随机噪声
    array = np.array(image)
    noise = np.random.randint(5, 95, array.shape, dtype='uint8')
    array += noise
    array = np.clip(array, 0, 255)
    image = Image.fromarray(array)
    
    # 随机裁剪
    width, height = image.size
    left = random.randint(0, width // 4)
    top = random.randint(0, height // 4)
    right = random.randint(3 * width // 4, width)
    bottom = random.randint(3 * height // 4, height)
    image = image.crop((left, top, right, bottom))
    
    # 随机拉伸
    new_width = random.randint((width // 2), width * 2)
    new_height = random.randint((height // 2), height * 2)
    image = image.resize((new_width, new_height))
    
    # 颜色调整
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))
    
    # 添加模糊效果
    image = image.filter(ImageFilter.GaussianBlur(random.randint(1, 3)))

    return image

# 对图像文件夹中的每个图像应用数据增强
for img_file in tqdm(os.listdir(image_folder)):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path)

        # 为每张图片生成约390个增强版本
        for i in range(10):
            aug_image = augment_image(image)
            augmented_image_path = os.path.join(augmented_folder, f'aug_{i}_{img_file}')
            if aug_image.mode == 'RGBA':
                aug_image = aug_image.convert('RGB')
            aug_image.save(augmented_image_path)

print("数据增强完成。")