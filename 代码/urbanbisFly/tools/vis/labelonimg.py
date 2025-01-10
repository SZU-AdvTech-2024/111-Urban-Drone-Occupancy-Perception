import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
from skimage.transform import resize
"""
将label 点映到图片上
"""
# 文件路径
num = "1727591176703"
pathcoor = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depthAndSeg\\"+num+"_coor_.pkl"
pathlabel = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depthAndSeg\\"+num+"_label_.pkl"
pathdepth = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depthAndSeg\\"+num+"_depthvalues_.pkl"

image_np = np.array(Image.open("E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\imgs\\img_SimpleFlight_0_0_"+num+".png"))
# image_np = resize(image_np, (400, 750), anti_aliasing=True)

# 加载数据
with open(pathcoor,"rb")  as f:
    coor = pickle.load(f)[0]
with open(pathlabel,"rb")  as f:
    label = pickle.load(f)[0]
with open(pathdepth,"rb")  as f:
    depth = pickle.load(f)[0]

depth_path = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depthAndSeg\\"+num+".npy"
depths = np.load(depth_path)[0]
img_clipped = np.clip(depths, 0, 255)

# 将图像转换为8位（0-255）以显示
img_8bit = img_clipped.astype(np.uint8)

# 显示裁剪后的图像
cv2.imshow('PFM Image (Clipped)', img_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 定义标签到颜色的映射
# 使用7种颜色，每种颜色对应一个标签 (1-7)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
cmap = ListedColormap(colors)

# 创建图像显示
fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
ax.imshow(image_np)

# 在图像上绘制标记点
for (x, y), lbl in zip(coor, label):
    color = cmap(lbl - 1)  # 标签减1以匹配颜色映射索引
    ax.plot(x, y, 'o', color=color, markersize=5)  # 绘制点

# 移除坐标轴并显示标记后的图像
ax.axis("off")
plt.show()
