import os
import random

def delete_half_of_images(folder_path):
    # 获取文件夹中的所有文件
    files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # 计算要删除的文件数
    num_files_to_delete = len(files) // 2

    # 随机选择并删除文件
    for file in random.sample(files, num_files_to_delete):
        os.remove(os.path.join(folder_path, file))
        print(f'Deleted {file}')

# 使用示例
folder_path = 'augmented_mcdd'  # 替换为你的文件夹路径
delete_half_of_images(folder_path)