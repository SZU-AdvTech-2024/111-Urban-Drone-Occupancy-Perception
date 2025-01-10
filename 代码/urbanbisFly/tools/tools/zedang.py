import numpy as np
import open3d as o3d

def point_cloud_to_depth_image_np(points, img_width, img_height, fov=90):
    """
    将点云投影到图像平面并生成深度图，FOV 为 90°。

    :param points: 形状为 (N, 3) 的点云数据 (x, y, z)。
    :param img_width: 图像宽度。
    :param img_height: 图像高度。
    :param fov: 视角 (默认 90°)。
    :return: 生成的深度图，形状为 (img_height, img_width)。
    """
    # 计算相机焦距，根据 FOV = 90° 的情况
    tan_half_fov = np.tan(np.radians(fov / 2.0))
    # fx = img_width / (2 * tan_half_fov)
    # fy = img_height / (2 * tan_half_fov)
    # fy = 375
    # # 主点在图像中心
    # cx = img_width / 2.0
    # cy = img_height / 2.0

    # 提取点云坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 过滤掉 z <= 0 的点，因为它们在相机后方
    valid_mask = z > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # 投影到图像平面，计算像素坐标 u, v
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # 创建深度图并初始化为无穷大（表示没有对应的深度）
    depth_image = np.full((img_height, img_width), np.inf)

    # 将深度值放入对应的像素坐标 (u, v)
    for i in range(len(u)):
        if 0 <= u[i] < img_width and 0 <= v[i] < img_height:
            depth_image[v[i], u[i]] = min(depth_image[v[i], u[i]], z[i])

    return depth_image
def depth_to_point_cloud(depth_image):
    """
    将深度图转换为点云。

    :param depth_image: 深度图，大小为 (h, w)，每个像素包含深度值。
    :param K: 相机内参矩阵 (3x3)。
    :return: 点云 (n, 3)，每个点的 3D 坐标 (X, Y, Z)。
    """
    h, w = depth_image.shape

    # 获取内参矩阵的参数
    # fx = 375
    # fy = 375
    # cx = 375
    # cy = 200

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 深度值
    Z = depth_image

    # 计算反投影的 X, Y 坐标
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # 创建一个掩码，去掉无效的深度值
    valid_mask = np.isfinite(Z)  # 只保留有限的深度值

    # 使用掩码筛选有效的 X, Y, Z 坐标
    X = X[valid_mask]
    Y = Y[valid_mask]
    Z = Z[valid_mask]
    # 将 X, Y, Z 组合成点云 (n, 3)
    point_cloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    # np.savetxt("test.txt",point_cloud,delimiter=',')
    return point_cloud
def euler_to_rotation_matrix(angles):
    """Convert Euler angles to rotation matrix."""
    rx, ry, rz = np.deg2rad(angles)

    cos_rx, cos_ry, cos_rz = np.cos([rx, ry, rz])
    sin_rx, sin_ry, sin_rz = np.sin([rx, ry, rz])

    R_x = np.array([
        [1, 0, 0],
        [0, cos_rx, -sin_rx],
        [0, sin_rx, cos_rx]
    ])

    R_y = np.array([
        [cos_ry, 0, sin_ry],
        [0, 1, 0],
        [-sin_ry, 0, cos_ry]
    ])

    R_z = np.array([
        [cos_rz, -sin_rz, 0],
        [sin_rz, cos_rz, 0],
        [0, 0, 1]
    ])
    # rx, ry, rz =np.deg2rad(np.array([0, 0, 90]))
    # cos_rx, cos_ry, cos_rz = np.cos([rx, ry, rz])
    # sin_rx, sin_ry, sin_rz = np.sin([rx, ry, rz])
    # R_x1 = np.array([
    #     [1, 0, 0],
    #     [0, cos_rx, -sin_rx],
    #     [0, sin_rx, cos_rx]
    # ])
    return R_z @ R_y @ R_x
def create_homogeneous_matrix(translation, rotation):
    """Create a homogeneous transformation matrix."""
    R = euler_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T
def inverse_rigid_transform(T):
    # 提取旋转矩阵 R 和平移向量 t
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    # 计算逆矩阵
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)

    # 构造逆的刚性变换矩阵
    T_inv = np.identity(4)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv

    return T_inv


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

# 配置文件中的参数

width = 1500
height = 800
fov_degrees = 90

# 计算焦距
fov_radians = np.deg2rad(fov_degrees)
# 计算水平焦距 fx
fx = width / (2 * np.tan(fov_radians / 2))

# 根据宽高比计算垂直视场角 FOV_v
FOV_v_rad = 2 * np.arctan((height / width) * np.tan(fov_radians / 2))

# 计算垂直焦距 fy
fy = height / (2 * np.tan(FOV_v_rad / 2))

# 计算图像中心坐标
cx = width / 2
cy = height / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
# 读取点云
points_txt = "E:\project\\urbanbisFly\\tools\out\8.txt"

depth_maps = []
for i in range(0,5):
    matrix = create_homogeneous_matrix(TS[i],RS[i])
    matrix = inverse_rigid_transform(matrix)
    point_cloud = np.loadtxt(points_txt, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    # point_cloud = voxelize_onlypoints(point_cloud,1)
    # points = apply_transformation(points,matrix)
    R = matrix[0:3,0:3]
    T = matrix[0:3, 3].reshape(3,1)
    a = np.hstack((R, T))
    P = K @ a
    # 将点云从齐次坐标转换到3D齐次坐标
    points_3D_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    A = points_3D_hom@(a.T)

    depth_image = point_cloud_to_depth_image_np(A,width,height)
    point = depth_to_point_cloud(depth_image)
    # 将 NumPy 数组转换为 Open3D 的 PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)

    # 假设相机的位置在 (0, 0, 0)，你可以根据实际情况调整
    camera_position = np.array([0, 0, 0])

    # 使用 Open3D 的 hidden_point_removal 进行遮挡剔除
    radius = 20000  # 设置一个合适的半径
    _, pt_map = pcd.hidden_point_removal(camera_position, radius)

    # 提取可见点
    visible_pcd = pcd.select_by_index(pt_map)

    # 将可见点的 NumPy 数组提取出来
    visible_points_np = np.asarray(visible_pcd.points)

    # 保存可见点为 .txt 文件
    output_txt_path = "visible_point_cloud"+str(i)+".txt"
    np.savetxt(output_txt_path, visible_points_np, fmt="%.6f", delimiter=" ")

    # 打印可见点的数量
    print(f"原始点的数量: {len(point)}, 可见点的数量: {len(visible_points_np)}")
    print(f"可见点已保存到: {output_txt_path}")
