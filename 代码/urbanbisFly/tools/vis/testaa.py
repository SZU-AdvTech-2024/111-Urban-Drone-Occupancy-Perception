import numpy as np

pt_path = 'D:\desk\\vcc\\save.pt'
import torch
import torch.nn.functional as F

# 假设创建一个形状为(6, 800, 1500)的示例张量（这里随机初始化）
gt_depth = torch.randn(6, 800, 1500)

# 使用interpolate函数进行插值操作，将其调整为(6, 512, 1408)的形状
interpolated_gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(512, 1408), mode='bilinear', align_corners=True).squeeze(0)

print(interpolated_gt_depth.shape)  # 输出为torch.Size([6, 512, 1408])
import cv2
import torch
import torch
# depth_map = torch.load(pt_path)
# depth = np.array(depth_map[0])
# img_clipped = np.clip(depth, 0, 255)
#
# # 将图像转换为8位（0-255）以显示
# img_8bit = img_clipped.astype(np.uint8)
#
# # 显示裁剪后的图像
# cv2.imshow('PFM Image (Clipped)', img_8bit)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(11)
def grid_to_point_cloud(grid):
    points = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                if grid[i,j,k]!= 0 and grid[i,j,k]!= 17:
                    points.append([i,j,k,grid[i][j][k]])
    return np.array(points)
mask_lidar_path = "E:\project\\urbanbisFly\mask_lidar.npy"
mask_camera_path = "E:\project\\urbanbisFly\mask_camera.npy"
voxel = "E:\project\\urbanbisFly\\voxel_semantics.npy"
a = np.load(mask_lidar_path)
a = grid_to_point_cloud(a)
b = np.load(mask_camera_path)
b = grid_to_point_cloud(b)
c = np.load(voxel)
c = grid_to_point_cloud(c)
np.savetxt("lidar.txt",a,delimiter=',')
np.savetxt("camera.txt",b,delimiter=',')
np.savetxt("voxel.txt",c,delimiter=',')
print(11)