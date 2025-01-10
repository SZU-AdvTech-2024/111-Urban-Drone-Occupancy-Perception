import math

import numpy as np
import cv2

from tools.toUE import create_homogeneous_matrix

# depth_path = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depth\\1727590894535.npy"
depth_path = "E:\project\\urbanbisFly\photos\scene\depth\depth.npy"
depths = np.load(depth_path)
img_clipped = np.clip(depths, 0, 255)

# 将图像转换为8位（0-255）以显示
img_8bit = img_clipped.astype(np.uint8)

# 显示裁剪后的图像
cv2.imshow('PFM Image (Clipped)', img_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 裁剪图像值，超过255的值设置为255
for i in depths:
    img_clipped = np.clip(i, 0, 255)

    # 将图像转换为8位（0-255）以显示
    img_8bit = img_clipped.astype(np.uint8)

    # 显示裁剪后的图像
    cv2.imshow('PFM Image (Clipped)', img_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


TS = [np.array([0, 0, 0]),
      np.array([0, 0, 0]),
      np.array([0, 0, 0]),
      np.array([0, 0, 0]),
      np.array([0, 0, 0])]
RS = [np.array([-60, 0, 0]),
      np.array([-60, -90, 0]),
      np.array([-60, 90, 0]),
      np.array([-90, 0, 0]),
      np.array([-60, -180, 0])]
Width = 1500
Height = 800
CameraFOV = 90
Colour = (0, 255, 0)
RGB = "%d %d %d" % Colour # Colour for points
points_clouds = []
def savepointcloud(image, filename):
    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0]!=0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
    f.close()
def depth_world(depth,filename):
    Fx = Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
    Cx = Width / 2
    Cy = Height / 2
    #转numpy
    # depth = np.array(depth.image_data_float, dtype=np.float64)
    depth[depth > 255] = 0
    rows, cols = depth.shape
    #2d->3d(内参)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # z = 100 * np.where(valid, depth / 256.0, np.nan)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - Cx) / Fx, 0)
    y = np.where(valid, z * (r - Cy) / Fy, 0)
    point = np.dstack((x, y, z))

    #相机外参 平移 绝对值
    savepointcloud(point,filename)
path = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-14-21-30\depth\\1727590893272.npy"
a = np.load(path)
img_clipped = np.clip(a, 0, 255)
#
# # 将图像转换为8位（0-255）以显示
# img_8bit = img_clipped.astype(np.uint8)
#
# # 显示裁剪后的图像
# cv2.imshow('PFM Image (Clipped)', img_8bit)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# depth_world(np.load(path),"test.asc")

# for i in range(len(depths)):
#     depth_world(depths[i],"test"+str(i)+".asc")
