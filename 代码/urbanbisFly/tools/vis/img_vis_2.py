import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time

# 定义文件夹路径
folder_path = 'E:\dataset\\urbanbis\ours\yuehai\\2024-11-14-10-39-41\imgs\\'  # 当前文件夹，你可以替换成实际的文件夹路径

# 获取所有符合条件的图片文件
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("img_SimpleFlight_0_0")]

# 创建图形对象和坐标轴对象
fig, ax = plt.subplots()

# 遍历图片并显示
for image_file in image_files:
    image = imread(image_file)
    ax.imshow(image)
    ax.axis('off')
    plt.pause(0.1)
    fig.canvas.draw()
    fig.canvas.flush_events()