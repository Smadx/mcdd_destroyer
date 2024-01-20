import requests
from bs4 import BeautifulSoup
import os
import re

# 爬取的网站
url = 'https://zhuanlan.zhihu.com/p/447084696'

# 发送请求
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 创建存储图片的文件夹
if not os.path.exists('images'):
    os.makedirs('images')

# 查找所有图片链接
for img_tag in soup.find_all('img'):
    img_url = img_tag.get('src')
    # 确保链接是JPG格式的图片
    if img_url and '.jpg' in img_url:
        # 清理文件名中的非法字符
        clean_file_name = re.sub(r'\?.*$', '', img_url.split('/')[-1])
        # 下载图片
        img_data = requests.get(img_url).content
        file_name = os.path.join('images', clean_file_name)
        with open(file_name, 'wb') as file:
            file.write(img_data)
        print(f'图片已下载: {file_name}')