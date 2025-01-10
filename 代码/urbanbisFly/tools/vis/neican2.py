import math
import re
import cv2
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
Colour = (0, 255, 0)
RGB = "%d %d %d" % Colour # Colour for points
def savepointcloud(image, filename):

    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0]!=0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
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

def pix23d(depth,save_name,Width,Height):
    Fx = Fy = Width / (2 * math.tan(90 * math.pi / 360))
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
def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]
def read_asc_to_numpy(file_path):
    """
    Reads a .asc point cloud file and converts it to a NumPy array.

    Args:
        file_path (str): The path to the .asc file.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3) where N is the number of points.
    """
    points = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.split(",")
            x, y, z = map(float, data[:3])
            points.append([x, y, z])

    return np.array(points)
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


def enhanced_point_cloud_filter(points, img_width, img_height, max_depth=1e6, search_radius=2):
    """
    改进后的点云投影与遮挡剔除算法，使用可调节的搜索半径来进行遮挡判断，
    并将深度图中的无穷大 (inf) 替换为一个很大的数字。

    :param points: 点云数据，形状为 (N, 3)
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param fov: 视角 (默认 90°)
    :param max_depth: 用于替换 inf 的最大深度值
    :param search_radius: 搜索半径，默认值为 2，表示检查周围 2 个像素内的点
    :return: 剔除遮挡后的点云
    """
    # 焦距计算
    # tan_half_fov = np.tan(np.radians(fov / 2.0))
    # fx = img_width / (2 * tan_half_fov)
    # fy = img_height / (2 * tan_half_fov)
    # cx = img_width / 2.0
    # cy = img_height / 2.0

    # 提取点云坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 只保留相机前方的点
    valid_mask = z > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # 投影到图像平面
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    # 创建深度图并初始化为无穷大
    depth_image = np.full((img_height, img_width), np.inf)

    # 替换 inf 为 max_depth
    depth_image[np.isinf(depth_image)] = max_depth

    # 遮挡剔除
    filtered_points = []
    for i in range(len(u)):
        if 0 <= u[i] < img_width and 0 <= v[i] < img_height:
            # 搜索半径内的最小深度
            min_depth = max_depth

            # 扩展搜索半径，检查邻域像素
            for du in range(-search_radius, search_radius + 1):
                for dv in range(-search_radius, search_radius + 1):
                    uu = u[i] + du
                    vv = v[i] + dv
                    if 0 <= uu < img_width and 0 <= vv < img_height:
                        # 检查邻域深度并更新最小深度
                        min_depth = min(min_depth, depth_image[vv, uu])

            # 如果当前点深度小于周围最小深度，则更新深度图
            if z[i] < min_depth:
                depth_image[v[i], u[i]] = z[i]  # 只更新当前点
                filtered_points.append(points[valid_mask][i])  # 保存未被遮挡的点

    return np.array(filtered_points)

def Cam2pix(TS, RS, point_cloud):
    # 计算焦距
    fov_radians = np.deg2rad(fov_degrees)
    fx = width / (2 * np.tan(fov_radians / 2))
    fy = height / (2 * np.tan(fov_radians / 2))

    # 计算光心
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                  [0, fx, cy],
                  [0, 0, 1]])
    pixs = []
    matrixs = []
    for i in range(0, 5):
        matrix = create_homogeneous_matrix(TS[i], RS[i])
        matrix = inverse_rigid_transform(matrix)
        # point_cloud = np.loadtxt(points_txt, delimiter=',', skiprows=1, usecols=(0, 1, 2))
        # points = apply_transformation(points,matrix)
        R = matrix[0:3, 0:3]
        T = matrix[0:3, 3].reshape(3, 1)
        a = np.hstack((R, T))
        P = K @ a
        # 将点云从齐次坐标转换到3D齐次坐标
        points_3D_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # 将3D点投影到2D图像平面
        points_2D_hom = P @ points_3D_hom.T
        points_2D = points_2D_hom[:2, :] / points_2D_hom[2, :]
        # points_2D = points_2D[:, points_2D.min(axis=0) >= 0]

        depth_image = np.full((height, width), np.inf)
        z_values = points_3D_hom[:, 2]  # 提取z值

        for i in range(points_2D.shape[1]):
            u, v = int(points_2D[0, i]), int(points_2D[1, i])

            # 确保像素坐标在图像范围内
            if 0 <= u < width and 0 <= v < height:
                depth_image[v, u] = min(depth_image[v, u], z_values[i])
        # 创建一个4x4的齐次变换矩阵
        homogeneous_matrix = np.zeros((4, 4))

        # 将3x4矩阵的值复制到4x4矩阵的前三行前四列
        homogeneous_matrix[:3, :4] = P

        # 设置齐次矩阵的右下角元素为1
        homogeneous_matrix[3, 3] = 1

        matrixs.append(homogeneous_matrix)
        depth_image[depth_image == np.inf] = 255
        img_clipped = np.clip(depth_image, 0, 255)

        # 将图像转换为8位（0-255）以显示
        img_8bit = img_clipped.astype(np.uint8)

        # 显示裁剪后的图像
        cv2.imshow('PFM Image (Clipped)', img_8bit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pixs.append(depth_image)
    return pixs, matrixs, K
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
def voxelize_onlypoints(points,  voxel_size):
    """
    将点云数据体素化并返回体素中心点的点云格式。

    参数:
    points (numpy.ndarray): 输入点云坐标数据，形状为 (n, 3)。
    labels (numpy.ndarray): 输入点云标签数据，形状为 (n,)。
    colors (numpy.ndarray): 输入点云颜色数据，形状为 (n, 3)。
    voxel_size (float): 体素的大小（边长）。

    返回:
    tuple: (voxel_center_points, voxel_labels, voxel_colors)
        voxel_center_points (numpy.ndarray): 体素中心点的坐标，形状为 (m, 3)。
        voxel_labels (numpy.ndarray): 体素中心点的标签，形状为 (m,)。
        voxel_colors (numpy.ndarray): 体素中心点的颜色，形状为 (m, 3)。
    """
    # 组合数据
    point_cloud = points

    # 计算体素坐标
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # 使用字典存储每个体素内的点
    voxel_dict = {}
    for point, idx in zip(point_cloud, voxel_indices):
        idx_tuple = tuple(idx)
        if idx_tuple not in voxel_dict:
            voxel_dict[idx_tuple] = []
        voxel_dict[idx_tuple].append(point)

    # 计算每个体素的中心点和属性的平均值
    voxel_center_points = []
    # voxel_labels = []
    # voxel_colors = []

    for voxel, points in voxel_dict.items():
        points_array = np.array(points)
        # voxel_center = (np.array(voxel) + 0.5) * voxel_size
        voxel_center = (np.array(voxel)) * voxel_size
        # 计算标签的众数
        # labels = points_array[:, 3].astype(int)
        # label_mode = np.bincount(labels).argmax()
        # # 计算颜色的平均值
        # colors_mean = points_array[:, 4:].mean(axis=0)

        voxel_center_points.append(voxel_center)
        # voxel_labels.append(label_mode + 1)  # 因为默认是0为忽略标签 所以要标签加1
        # voxel_colors.append(colors_mean)

    return np.array(voxel_center_points)
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

import numpy as np

def voxelize_point_cloud(point_cloud, voxel_size):
    # 假设点云是 Nx3 的 numpy 数组，包含点的 x, y, z 坐标
    voxel_indices = np.floor(point_cloud / voxel_size).astype(int)
    return voxel_indices


import numpy as np





import numpy as np



# 示例数据
reference_point = np.array([0, 0, 0])
point_cloud = np.array([
    [1.2, 1.2, 1.2],
    [2.5, 2.5, 2.5],
    [1.1, 2.3, 3.1],
    [1.8, 1.8, 1.8],
    [-1.5, -1.5, -1.5],
    [1.7, 1.7, 1.7],
])



def generate_ray_mask(point_cloud, reference_point, tolerance=1e-3):
    """
    对于点云中的每个点，生成一个 mask，标记哪些点与其他点在同一射线上。

    Parameters:
        point_cloud (numpy.ndarray): Nx3 的点云数组，表示每个点的 (x, y, z) 坐标。
        reference_point (numpy.ndarray): 相机的位置坐标，例如 [0, 0, 0]。
        tolerance (float): 容差，用于判断向量是否共线。

    Returns:
        numpy.ndarray: 形状为 NxN 的布尔 mask，True 表示两个点在同一射线上。
    """
    num_points = point_cloud.shape[0]

    # 计算每个点相对于参考点的方向向量
    directions = point_cloud - reference_point

    # 计算方向向量的模
    norms = np.linalg.norm(directions, axis=1)

    # 避免方向向量的模为0的情况（即点与参考点重合）
    valid = norms > tolerance
    directions[~valid] = 0  # 设置与参考点重合的点的方向向量为零向量

    # 归一化方向向量
    normalized_directions = directions / (norms[:, np.newaxis] + tolerance)

    # 批量计算点积，计算 NxN 的点积矩阵
    dot_products = np.dot(normalized_directions, normalized_directions.T)

    # 生成掩码：点积接近 1 或 -1 的点在同一射线上
    mask_matrix = np.abs(np.abs(dot_products) - 1) < tolerance

    # 删除对角线上的元素（自己与自己比较），因为它们始终是 True
    np.fill_diagonal(mask_matrix, False)

    # 对每个点，检查它是否与其他点在同一射线上
    mask = np.any(mask_matrix, axis=1)
    return mask
def ray_casting_occlusion_removal(point_cloud, camera_position, distance_threshold=1e-6):
    """
    使用光线投射剔除遮挡点云中的被遮挡点。

    Parameters:
        point_cloud (numpy.ndarray): Nx3 的点云数组，表示每个点的 (x, y, z) 坐标。
        camera_position (numpy.ndarray): 相机的位置坐标，例如 [0, 0, 0]。
        distance_threshold (float): 用于判断点在同一光线方向上的距离容差，防止浮点误差。

    Returns:
        numpy.ndarray: 剔除遮挡后的点云。
    """
    # 计算从相机到每个点的向量
    directions = point_cloud - camera_position
    # 计算归一化的方向向量
    normalized_directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # 将方向向量转换为tuple以便哈希比较
    direction_tuples = [tuple(direction) for direction in normalized_directions]

    # 存储每条光线上最接近相机的点
    closest_points = {}

    # 遍历每个点
    for i, direction in enumerate(direction_tuples):
        point = point_cloud[i]
        distance = np.linalg.norm(point - camera_position)

        # 如果这是该方向上第一个点，或者比当前已存的点更近，则更新
        if direction not in closest_points or closest_points[direction][1] > distance:
            closest_points[direction] = (point, distance)

    # 提取未被遮挡的点
    culled_points = np.array([closest_points[direction][0] for direction in closest_points])

    return culled_points


def keep_closest_points(point_cloud, voxel_indices, camera_position):
    # 创建字典来存储每个体素中最小深度的点
    closest_points = {}

    for i, point in enumerate(point_cloud):
        voxel = tuple(voxel_indices[i])
        depth = np.linalg.norm(point - camera_position)

        # 如果该体素还没有点，或者找到更近的点，则更新该体素的点
        if voxel not in closest_points or closest_points[voxel][1] > depth:
            closest_points[voxel] = (point, depth)

    # 提取最接近的点
    return np.array([closest_points[voxel][0] for voxel in closest_points])
def voxel_based_culling(point_cloud, voxel_size, camera_position):
    # Step 1: 体素化点云
    voxel_indices = voxelize_point_cloud(point_cloud, voxel_size)

    # Step 2: 保留每个体素中离相机最近的点
    culled_point_cloud = keep_closest_points(point_cloud, voxel_indices, camera_position)

    return culled_point_cloud

def remove_hidden_points(point_cloud, viewpoint=np.array([0,0,0]), radius=1000):
    """
    使用HPR算法移除被遮挡的点，并返回可见的点云。

    参数:
    point_cloud - numpy数组，表示点云 (N x 3)。
    viewpoint - 视点位置 (3,)。
    radius - 投影半径，通常设置为1.0。

    返回:
    可见点云。
    """
    # Step 1: 平移点云，使视点位于原点
    translated_points = point_cloud - viewpoint

    # Step 2: 将点云投影到球面上，R是投影半径
    projected_points = radius * translated_points / np.linalg.norm(translated_points, axis=1)[:, np.newaxis]

    # Step 3: 使用凸包找到可见的点
    hull = ConvexHull(projected_points)

    # Step 4: 获取凸包上可见点的索引
    visible_indices = hull.vertices

    # Step 5: 返回可见点云
    return point_cloud[visible_indices]
"""
读取文本中的相机坐标系 通过内外参转成2D深度图并可视化
"""
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
width = 750
height = 400
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
# K = \
#     np.array([[750, 0, 750],
#               [0, 809, 400.],
#               [0, 0., 1.]])
points_txt = "E:\project\\urbanbisFly\\tools\save.txt"
# points_txt = "E:\dataset\\urbanbis\ours\qingdao\\2024-09-30-14-43-22\pfm2cam\\pfm2cam_1727678605004.txt"
depth_maps = []
for i in range(0,5):
    matrix = create_homogeneous_matrix(TS[i],RS[i])
    matrix = inverse_rigid_transform(matrix)
    # point_cloud = np.loadtxt(points_txt, delimiter=',', skiprows=1, usecols=(0, 1, 2))
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
    np.savetxt(str(i) + ".txt", A , delimiter=',')
    # depth_image = point_cloud_to_depth_image_np(A,width,height)
    pixs, matrixs, K = Cam2pix(TS,RS,point_cloud)
    point = depth_to_point_cloud(depth_image)

    camera_position = np.array([0, 0, 0])
    # mask = generate_ray_mask(point,camera_position)
    # print(mask.sum().item())
    # point = voxelize_onlypoints(point, 1)


    # 将无穷大值替换为0或最大可视深度
    depth_image[depth_image == np.inf] = 255
    img_clipped = np.clip(depth_image, 0, 255)

    # 将图像转换为8位（0-255）以显示
    img_8bit = img_clipped.astype(np.uint8)

    # 显示裁剪后的图像
    cv2.imshow('PFM Image (Clipped)', img_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    depth_maps.append(depth_image)

    points_2D_hom = P @ points_3D_hom.T
    points_2D = points_2D_hom[:2, :] / points_2D_hom[2, :]
    # points_2D = points_2D[:, points_2D.min(axis=0) >= 0]
    # mask_h = np.abs(points_2D_hom[0, :]<=width)
    # mask_v = np.abs(points_2D_hom[1,:]<=height)
    # mask  =  mask_h & mask_v
    depth_image = np.full((height, width), np.inf)
    # A = points_3D_hom@(a.T)
    # np.savetxt(str(i)+".txt",A,delimiter=",")
    z_values = points_3D_hom[:, 2]  # 提取z值

    for i in range(points_2D.shape[1]):
        u, v = int(points_2D[0, i]), int(points_2D[1, i])

        # 确保像素坐标在图像范围内
        if 0 <= u < width and 0 <= v < height:
            depth_image[v, u] = min(depth_image[v, u], z_values[i])
    depth_image[depth_image == np.inf] = 255
    # 裁剪图像值，超过255的值设置为255
    # img_clipped = np.clip(depth_image, 0, 255)
    #
    # # 将图像转换为8位（0-255）以显示
    # img_8bit = img_clipped.astype(np.uint8)
    #
    # # 显示裁剪后的图像
    # cv2.imshow('PFM Image (Clipped)', img_8bit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # depth_maps.append(depth_image)

