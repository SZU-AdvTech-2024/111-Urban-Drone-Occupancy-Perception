import numpy as np

# # 假设点云数据存储在一个 N x 4 的 NumPy 数组中
# # 点云数据的列依次是 [x, y, z, label]
# point_cloud = np.random.randint(0, 96, size=(100, 4))  # 示例数据
path = "E:\dataset\\urbanbis\ours\yuehai\\2024-11-14-10-39-41\lidar_point\\lidar_point1731551983746.txt"
point_cloud = np.loadtxt(path,delimiter=',',skiprows=1)[:,:4]

# 获取所有的 x, y, z, label
x_values = point_cloud[:, 0]
y_values = point_cloud[:, 1]
z_values = point_cloud[:, 2]
labels = point_cloud[:, 3]

# 创建一个新的点云列表，存储降维后的点
new_point_cloud = []

# 获取所有不同的 (x, z) 对
unique_xz_pairs = np.unique(point_cloud[:, [0, 2]], axis=0)

# 遍历每个 (x, z) 对
for xz in unique_xz_pairs:
    # 找到所有 xz 对应的点
    mask = (x_values == xz[0]) & (z_values == xz[1])
    xz_points = point_cloud[mask]

    # 在这些点中，找到 y 值最大的点
    max_y_point = xz_points[np.argmax(xz_points[:, 1])]

    # 将该点的 (x, y, z, label) 添加到新的点云中
    new_point_cloud.append(max_y_point)

# 将新的点云转换为 NumPy 数组
new_point_cloud = np.array(new_point_cloud)
