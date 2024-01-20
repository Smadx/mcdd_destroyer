import os
import shutil
from PIL import Image

def convert_and_copy_images_to_jpg(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        target_path = os.path.join(target_folder, new_filename)

        if filename.lower().endswith('.jpg'):
            # 如果是JPG文件，直接复制
            shutil.copy2(source_path, target_path)
            print(f'Copied {filename} to {target_folder}')
        else:
            try:
                # 对于非JPG文件，先转换再保存
                with Image.open(source_path) as img:
                    img.convert('RGB').save(target_path)
                    print(f'Converted and copied {filename} to {new_filename} in {target_folder}')
            except IOError:
                print(f'Cannot process {filename}. Unsupported format or not an image.')

# 使用示例
source_folder = 'negative_samples'  # 替换为你的源文件夹路径
target_folder = 'negative_samples_jpg'  # 替换为你想保存新图片的文件夹路径
convert_and_copy_images_to_jpg(source_folder, target_folder)