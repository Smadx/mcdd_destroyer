import os

def check_for_non_jpg_images(folder_path):
    non_jpg_files = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.jpg'):
            non_jpg_files.append(filename)

    if non_jpg_files:
        print("Found non-JPG image files:")
        for file in non_jpg_files:
            print(file)
    else:
        print("No non-JPG image files found.")

# 使用示例
folder_path = 'negative_samples_jpg'  # 替换为你的文件夹路径
check_for_non_jpg_images(folder_path)