import cv2
import os
import re
import numpy as np
import math
'''
pfm文件可视化
'''

Colour = (0, 255, 0)
RGB = "%d %d %d" % Colour # Colour for points
depth = True
# 相机内参
Width = 750
Height = 400
CameraFOV = 90
def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:  # big-endian
            endian = '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        return np.reshape(data, shape)
def savepointcloud(image, filename):

    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0]!=0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
    f.close()
def pix23d(pfm_file_path,save_name,Width,Height):
    depth = read_pfm(pfm_file_path)
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
    points = np.dstack((x, y, z))
    savepointcloud(points, save_name)
    return points



def depth_world(depth,filename):


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
    # z = 100 * np.where(valid, depth / 256.0, np.nan)f
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - Cx) / Fx, 0)
    y = np.where(valid, z * (r - Cy) / Fy, 0)
    point = np.dstack((x, y, z))

    #相机外参 平移 绝对值
    savepointcloud(point,filename)


# # 指定PFM文件路径
pfm_file_path = 'E:\dataset\\urbanbis\data\lihu\\fortest\images\\img_SimpleFlight_0_1_1732622259911304800.pfm'

img = read_pfm(pfm_file_path)
print(11)
# #
depth_path = "test.asc"
depth_world(img,depth_path)
# #
# #
# # 裁剪图像值，超过255的值设置为255
img_clipped = np.clip(img, 0, 255)

# 将图像转换为8位（0-255）以显示
img_8bit = img_clipped.astype(np.uint8)

# 显示裁剪后的图像
cv2.imshow('PFM Image (Clipped)', img_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()
