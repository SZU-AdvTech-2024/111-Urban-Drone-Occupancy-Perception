import math

from PIL import Image
import os
import cv2
import numpy as np
imgs = ["img_SimpleFlight_0_0_1725534205188626400.png",
"img_SimpleFlight_1_0_1725534205190069400.png","img_SimpleFlight_2_0_1725534205191049300.png","img_SimpleFlight_3_0_1725534205192367500.png","img_SimpleFlight_4_0_1725534205193255800.png"]
# 指定图片文件夹路径
folder_path = 'C:\\Users\Admin\Documents\AirSim\\2024-09-05-19-03-24\images\\'

# 遍历文件夹中的所有文件
for filename in imgs:
    # 检查文件是否为图片
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 拼接图片的完整路径
        image_path = os.path.join(folder_path, filename)

        # 读取图片
        img = cv2.imread(image_path)

        # 显示图片
        cv2.imshow('Image', img)

        # 等待用户按下任意键后关闭窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def savepointcloud(image, filename):

    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0]!=0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
    f.close()
def depth_world(depth,filename):
    #相机内参
    Width = 1500
    Height = 800
    CameraFOV = 90

    Fx = Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
    Cx = Width / 2
    Cy = Height / 2
    #转numpy
    # depth = np.array(depth.image_data_float, dtype=np.float64)
    depth[depth > 255] = 255
    rows, cols = depth.shape
    #2d->3d(内参)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # z = 100 * np.where(valid, depth / 256.0, np.nan)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - Cx) / Fx, 0)
    y = np.where(valid, z * (r - Cy) / Fy, 0)
    point = np.dstack((x, y, z))

    #相机外参 平移 绝对值
    savepointcloud(point,filename)

# depth_file = "C:\\Users\Admin\Documents\AirSim\\2024-08-15-14-35-11\images\\img_SimpleFlight_0_1_1723703722404541800.png"
# img = cv2.imread(depth_file)
# depth_world(img,"test.asc")
