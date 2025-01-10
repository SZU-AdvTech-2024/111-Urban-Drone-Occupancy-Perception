"""
读取rotation和translation
由UE坐标系--》Mesh坐标系--》点云坐标系
1.获取点云并转成occ
2.将Mesh坐标系-》亚局部坐标系-》相机局部坐标系-》各自的原坐标系-》3D换2D拿到对应的深度图（考虑范围？）
"""
import math
import os.path
import re
import cv2
import numpy as np
import open3d as o3d

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pickle
from tools.tools.angle import euler_to_homogeneous_matrix

Colour = (0, 255, 0)

RGB = "%d %d %d" % Colour  # Colour for points

# lidar2cams = [
#     np.array([[1., 0., 0., 0.],
#               [0., -0.6427876, -0.7660444, 0.],
#               [0., 0.7660444, -0.6427876, 0.],
#               [0., 0., 0., 1.]])
#     ,
#     np.array([[-1., 0., 0., 0.],
#               [0., 0.6427876, 0.7660444, 0.],
#               [0., 0.7660444, -0.6427876, 0.],
#               [0., 0., 0., 1.]])
#     ,
#     np.array([[0., -1., 0., 0.],
#               [0.7660444, 0., -0.6427876, 0.],
#               [0.6427876, 0., 0.7660444, 0.],
#               [0., 0., 0., 1.]])
#     ,
#     np.array([[0., 1., 0., 0.],
#               [-0.7660444, 0., 0.6427876, 0.],
#               [-0.6427876, 0., -0.7660444, 0.],
#               [0., 0., 0., 1.]])
#     ,
#     np.array([[1., 0., 0., 0.],
#               [0., 0., 1., 0.],
#               [0., -1., 0., 0.],
#               [0., 0., 0., 1.]])
# ]
# intrinsic = \
#     np.array([[750, 0, 750],
#               [0, 809, 400.],
#               [0, 0., 1.]])

def point_cloud_to_depth_image(point_cloud, intrinsic):
    """
    将3D点云投影到深度图上。

    参数:
    - point_cloud: np.array，形状为 (N, 3)，每行是一个3D点 (x, y, z)。

    返回:
    - depth_image: np.array，形状为 (image_height, image_width)，每个像素值表示深度。
    """
    image_width = depth_width
    image_height = depth_height
    # 定义相机内参矩阵
    intrinsic_matrix = intrinsic

    # 仅保留相机前方的点（z > 0）
    valid_points = point_cloud[:, 2] > 0
    point_cloud = point_cloud[valid_points]

    # 使用内参矩阵将点投影到图像平面
    projected_points = intrinsic_matrix @ point_cloud.T  # (3, N)
    projected_points = projected_points[:2, :] / projected_points[2, :]  # 透视除法，归一化

    # 将投影后的坐标转换为最近的像素
    u_coords = np.round(projected_points[0, :]).astype(int)
    v_coords = np.round(projected_points[1, :]).astype(int)
    depth_values = point_cloud[:, 2]  # 深度值（z坐标）

    # 初始化深度图
    depth_image = np.full((image_height, image_width), np.inf)  # 初始化为inf深度
    valid_indices = (u_coords >= 0) & (u_coords < image_width) & (v_coords >= 0) & (v_coords < image_height)
    u_coords = u_coords[valid_indices]
    v_coords = v_coords[valid_indices]
    depth_values = depth_values[valid_indices]

    # 填充深度图，保留每个像素的最近深度值
    for u, v, depth in zip(u_coords, v_coords, depth_values):
        if depth < depth_image[v, u]:  # 仅保留最近点（最小深度）
            depth_image[v, u] = depth

    # 将无穷深度（背景）值替换为0
    depth_image[depth_image == np.inf] = 0

    return depth_image


def point_cloud_to_depth_image_with_labels(point_cloud, intrinsic, labels):
    """
    将3D点云投影到深度图上并记录像素坐标、标签和深度。

    参数:
    - point_cloud: np.array，形状为 (N, 3)，每行是一个3D点 (x, y, z)。
    - intrinsic: np.array，相机内参矩阵。
    - labels: np.array，形状为 (N,)，每个点对应的标签。
    - image_width: int，深度图的宽度。
    - image_height: int，深度图的高度。

    返回:
    - depth_image: np.array，形状为 (image_height, image_width)，每个像素值表示深度。
    - coor: list，包含有效深度图像素位置的坐标 (u, v)。
    - output_labels: list，包含每个有效像素的标签。
    - depth_values_output: list，包含每个有效像素的深度值。
    """
    # 定义相机内参矩阵
    intrinsic_matrix = intrinsic

    # 仅保留相机前方的点（z > 0）
    valid_points = point_cloud[:, 2] > 0
    point_cloud = point_cloud[valid_points]
    labels = labels[valid_points]

    # 使用内参矩阵将点投影到图像平面
    projected_points = intrinsic_matrix @ point_cloud.T  # (3, N)
    projected_points = projected_points[:2, :] / projected_points[2, :]  # 透视除法，归一化

    # 将投影后的坐标转换为最近的像素
    u_coords = np.round(projected_points[0, :]).astype(int)
    v_coords = np.round(projected_points[1, :]).astype(int)
    depth_values = point_cloud[:, 2]  # 深度值（z坐标）

    # 初始化深度图
    depth_image = np.full((depth_height, depth_width), np.inf)  # 初始化为inf深度
    valid_indices = (u_coords >= 0) & (u_coords < depth_width) & (v_coords >= 0) & (v_coords < depth_height)
    u_coords = u_coords[valid_indices]
    v_coords = v_coords[valid_indices]
    depth_values = depth_values[valid_indices]
    labels = labels[valid_indices]

    # 用于存储有效像素的坐标、标签和深度值
    coor = []
    output_labels = []
    depth_values_output = []

    # 填充深度图，保留每个像素的最近深度值
    for u, v, depth, label in zip(u_coords, v_coords, depth_values, labels):
        if depth < depth_image[v, u]:  # 仅保留最近点（最小深度）
            depth_image[v, u] = depth
            coor.append((u, v))           # 记录像素位置
            output_labels.append(label)    # 记录对应的标签
            depth_values_output.append(depth)  # 记录对应的深度值

    # 将无穷深度（背景）值替换为0
    depth_image[depth_image == np.inf] = 0

    return depth_image, coor, output_labels, depth_values_output
def to_eularian_angles(Q_W, Q_X, Q_Y, Q_Z):
    z = Q_W
    y = Q_X
    x = Q_Y
    w = Q_Z
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return pitch, roll, yaw


def quaternion_to_euler(Q_W, Q_X, Q_Y, Q_Z):
    r = R.from_quat([Q_W, Q_X, Q_Y, Q_Z])
    euler_angles = r.as_euler('xyz')  # 'xyz' 指定旋转顺序，degrees=True 表示输出角度值

    return euler_angles[0], euler_angles[1], euler_angles[2]


def icp_registration(point_cloud_A_np, point_cloud_B_np, voxel_size=0.05, threshold=0.02):
    """
    使用ICP算法将点云B配准到点云A。

    参数:
    - point_cloud_A_np: numpy格式的点云A，形状为 (N, 3)
    - point_cloud_B_np: numpy格式的点云B，形状为 (M, 3)
    - voxel_size: 用于特征匹配的体素大小，影响特征的计算和初步对齐
    - threshold: ICP算法的最近点距离阈值，影响配准精度

    返回:
    - aligned_point_cloud_B: numpy格式的配准后的点云B
    - transformation: 从B到A的变换矩阵 (4x4)
    """

    # 将 numpy 数组转换为 open3d 点云
    point_cloud_A = o3d.geometry.PointCloud()
    point_cloud_B = o3d.geometry.PointCloud()
    point_cloud_A.points = o3d.utility.Vector3dVector(point_cloud_A_np)
    point_cloud_B.points = o3d.utility.Vector3dVector(point_cloud_B_np)

    # 计算法向量
    point_cloud_A.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    point_cloud_B.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 计算FPFH特征
    fpfh_A = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud_A,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    fpfh_B = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud_B,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    # 使用特征匹配估计初始变换
    def feature_matching(source, target, source_fpfh, target_fpfh):
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            num_iterations=1000
        )
        return result.transformation

    trans_init = feature_matching(point_cloud_B, point_cloud_A, fpfh_B, fpfh_A)

    # 使用点到面的ICP配准
    reg_icp = o3d.pipelines.registration.registration_icp(
        point_cloud_B, point_cloud_A, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # 打印配准信息
    print("ICP收敛:", reg_icp.convergence_criteria)
    print("最终变换矩阵:\n", reg_icp.transformation)

    # 应用配准变换
    point_cloud_B.transform(reg_icp.transformation)

    # 将配准后的点云B转换回numpy格式
    aligned_point_cloud_B = np.asarray(point_cloud_B.points)
    return aligned_point_cloud_B
    # return aligned_point_cloud_B, reg_icp.transformation
def read_txtdata(path):
    df = pd.read_csv(path, sep='\t')
    #四元组-》欧拉角
    euler_angles = df.apply(lambda row: to_eularian_angles(row['Q_W'], row['Q_X'], row['Q_Y'], row['Q_Z']), axis=1)
    df[['Roll', 'Pitch', 'Yaw']] = pd.DataFrame(euler_angles.tolist(), index=df.index)
    return df.to_dict(orient='records')


def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    voxel_centers = data[:, :3]
    labels = data[:, 3].astype(int)
    colors = data[:, 4:7]
    return voxel_centers, labels, colors


def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

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

def inverse_rigid_transform(matrix):
    """
    Computes the inverse of a 4x4 rigid transformation matrix.

    Parameters:
    matrix (np.ndarray): A 4x4 rigid transformation matrix.

    Returns:
    np.ndarray: The inverse of the given 4x4 rigid transformation matrix.
    """
    assert matrix.shape == (4, 4), "Input matrix must be a 4x4 matrix"

    # Create an identity matrix for the inverse
    inverse_matrix = np.eye(4)

    # Transpose the 3x3 rotation part
    inverse_matrix[:3, :3] = matrix[:3, :3].T

    # Adjust the translation part
    inverse_matrix[:3, 3] = -np.dot(inverse_matrix[:3, :3], matrix[:3, 3])

    return inverse_matrix


def filter_voxels_in_box(voxel_centers, labels, colors, center_point, x_range, y_range, z_range, voxel_size):
    min_x, max_x = center_point[0][0] - x_range[0], center_point[0][0] + x_range[1]
    min_y, max_y = center_point[0][1] - y_range[0], center_point[0][1] + y_range[1]
    min_z, max_z = center_point[0][2] - z_range[0], center_point[0][2] + z_range[1]

    in_box = (
            ((voxel_centers[:, 0] >= (min_x * voxel_size)) & (voxel_centers[:, 0] <= (max_x * voxel_size))) &
            ((voxel_centers[:, 1] >= (min_y * voxel_size)) & (voxel_centers[:, 1] <= (max_y * voxel_size))) &
            ((voxel_centers[:, 2] >= (min_z * voxel_size)) & (voxel_centers[:, 2] <= (max_z * voxel_size)))
    )

    filtered_centers = voxel_centers[in_box]
    filtered_labels = labels[in_box]
    filtered_colors = colors[in_box]

    return filtered_centers, filtered_labels, filtered_colors


def voxelize(points, labels, colors, voxel_size):
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
    point_cloud = np.hstack((points, labels[:, np.newaxis], colors))

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
    voxel_labels = []
    voxel_colors = []

    for voxel, points in voxel_dict.items():
        points_array = np.array(points)
        # voxel_center = (np.array(voxel) + 0.5) * voxel_size
        voxel_center = (np.array(voxel)) * voxel_size
        # 计算标签的众数
        labels = points_array[:, 3].astype(int)
        label_mode = np.bincount(labels).argmax()
        # 计算颜色的平均值
        colors_mean = points_array[:, 4:].mean(axis=0)

        voxel_center_points.append(voxel_center)
        voxel_labels.append(label_mode + 1)  # 因为默认是0为忽略标签 所以要标签加1
        voxel_colors.append(colors_mean)

    return np.array(voxel_center_points), np.array(voxel_labels), np.array(voxel_colors)

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
def voxelize_nolabel(points, voxel_size):
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
    # point_cloud = np.hstack((points, labels[:, np.newaxis], colors))

    # 计算体素坐标
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # 使用字典存储每个体素内的点
    voxel_dict = {}
    for point, idx in zip(points, voxel_indices):
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
        # # 计算标签的众数
        # labels = points_array[:, 3].astype(int)
        # label_mode = np.bincount(labels).argmax()
        # # 计算颜色的平均值
        # colors_mean = points_array[:, 4:].mean(axis=0)

        voxel_center_points.append(voxel_center)
        # voxel_labels.append(label_mode + 1)  # 因为默认是0为忽略标签 所以要标签加1
        # voxel_colors.append(colors_mean)

    return np.array(voxel_center_points)
def save_filtered_voxel_data_as_txt(voxel_centers, labels, colors, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z,label,r,g,b\n")
        for center, label, color in zip(voxel_centers, labels, colors):
            f.write(f"{center[0]},{center[1]},{center[2]},{label},{color[0]},{color[1]},{color[2]}\n")
def save_filtered_voxel_data_as_txt_nocolor(voxel_centers, labels, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z,label\n")
        for center, label in zip(voxel_centers, labels):
            f.write(f"{center[0]},{center[1]},{center[2]},{label}\n")

def save_txt(voxel_centers, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z\n")
        for center in voxel_centers:
            f.write(f"{center[0]},{center[1]},{center[2]}")


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


def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]


def create_homogeneous_matrix(translation, rotation):
    """Create a homogeneous transformation matrix."""
    R = euler_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def savepointcloud(image, filename):
    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0] != 0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
    f.close()


def generate_variants(point,fanwei):
    variants = []

    # 单个坐标的加减
    for i in range(3):
        for delta in fanwei:
            new_point = list(point[:3])
            new_point[i] += delta
            variants.append(tuple(new_point))

    # 两个坐标的加减
    # for i in range(3):
    #     for j in range(i + 1, 3):
    #         for delta in fanwei:
    #             new_point = list(point[:3])
    #             new_point[i] += delta
    #             new_point[j] += delta
    #             variants.append(tuple(new_point))

    # 三个坐标的加减
    for delta in fanwei:
        new_point = list(point[:3])
        new_point[0] += delta
        new_point[1] += delta
        new_point[2] += delta
        variants.append(tuple(new_point))

    return variants
def HPR(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 假设相机的位置在 (0, 0, 0)，你可以根据实际情况调整
    camera_position = np.array([0, 0, 0])

    # 使用 Open3D 的 hidden_point_removal 进行遮挡剔除
    radius = 20000  # 设置一个合适的半径
    _, pt_map = pcd.hidden_point_removal(camera_position, radius)

    # 提取可见点
    visible_pcd = pcd.select_by_index(pt_map)

    # 将可见点的 NumPy 数组提取出来
    visible_points_np = np.asarray(visible_pcd.points)
    return visible_points_np

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
            data = line.split()
            x, y, z = map(float, data[:3])
            points.append([x, y, z])

    return np.array(points)

def pix23d(pfm_file_path):
    depth = read_pfm(pfm_file_path)

    fov_degrees = 90
    depth_width = 750
    depth_height = 400
    fx = fy = depth_width / (2 * np.tan(np.radians(fov_degrees) / 2))
    cx, cy = depth_width / 2, depth_height / 2
    intrinsic = intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    Fx = fx
    Fy = fy
    Cx = cx
    Cy = cy
    # 转numpy
    # depth = np.array(depth.image_data_float, dtype=np.float64)
    depth[depth > 255] = 255
    rows, cols = depth.shape
    # 2d->3d(内参)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # z = 100 * np.where(valid, depth / 256.0, np.nan)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - Cx) / Fx, 0)
    y = np.where(valid, z * (r - Cy) / Fy, 0)
    point = np.dstack((x, y, z))
    return point
def translation_to_homogeneous(translation):
    """
    Convert a 1x3 translation vector into a 4x4 homogeneous transformation matrix.

    Parameters:
    translation (list or np.ndarray): A 1x3 translation vector.

    Returns:
    np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    if len(translation) != 3:
        raise ValueError("Translation vector must have exactly 3 elements.")

    # Create a 4x4 identity matrix
    homogeneous_matrix = np.eye(4)

    # Set the translation vector in the homogeneous matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

def Cam2pix_matrixs_k(TS,RS):
    fov_radians = np.deg2rad(fov_degrees)
    K = intrinsic

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
        # 创建一个4x4的齐次变换矩阵
        homogeneous_matrix = np.zeros((4, 4))

        # 将3x4矩阵的值复制到4x4矩阵的前三行前四列
        homogeneous_matrix[:3, :4] = P

        # 设置齐次矩阵的右下角元素为1
        homogeneous_matrix[3, 3] = 1

        matrixs.append(homogeneous_matrix)
    return matrixs,K
def combine_matrices1(rigid_matrix, intrinsic_matrix):
    """
    组合4x4刚性变换矩阵和3x3内参矩阵，生成新的4x4矩阵。

    :param rigid_matrix: 4x4 刚性变换矩阵
    :param intrinsic_matrix: 3x3 内参矩阵
    :return: 4x4 组合后的矩阵
    """

    # 确保输入矩阵的形状正确
    assert rigid_matrix.shape == (4, 4), "刚性变换矩阵必须是4x4的"
    assert intrinsic_matrix.shape == (3, 3), "内参矩阵必须是3x3的"

    # 将内参矩阵扩展为4x4矩阵
    intrinsic_matrix_4x4 = np.eye(4)
    intrinsic_matrix_4x4[:3, :3] = intrinsic_matrix

    # 计算新的4x4矩阵
    combined_matrix = np.dot(intrinsic_matrix_4x4, rigid_matrix)

    return combined_matrix

def Cam2pix(TS, RS, point_cloud):
    # 计算焦距
    fov_radians = np.deg2rad(fov_degrees)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
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

        depth_image = np.full((depth_height, depth_width), np.inf)
        z_values = points_3D_hom[:, 2]  # 提取z值

        for i in range(points_2D.shape[1]):
            u, v = int(points_2D[0, i]), int(points_2D[1, i])

            # 确保像素坐标在图像范围内
            if 0 <= u < depth_width and 0 <= v < depth_height:
                depth_image[v, u] = min(depth_image[v, u], z_values[i])
        # 创建一个4x4的齐次变换矩阵
        homogeneous_matrix = np.zeros((4, 4))

        # 将3x4矩阵的值复制到4x4矩阵的前三行前四列
        homogeneous_matrix[:3, :4] = P

        # 设置齐次矩阵的右下角元素为1
        homogeneous_matrix[3, 3] = 1

        matrixs.append(homogeneous_matrix)
        pixs.append(depth_image)
    return pixs, matrixs, K
def point_cloud_to_depth_image_np(points ):
    """
    将点云投影到图像平面并生成深度图，FOV 为 90°。

    :param points: 形状为 (N, 3) 的点云数据 (x, y, z)。
    :param img_width: 图像宽度。
    :param img_height: 图像高度。
    :param fov: 视角 (默认 90°)。
    :return: 生成的深度图，形状为 (img_height, img_width)。
    """
    # 计算相机焦距，根据 FOV = 90° 的情况
    tan_half_fov = np.tan(np.radians(fov_degrees / 2.0))
    # fx = width / (2 * tan_half_fov)
    # fy = height / (2 * tan_half_fov)
    # fy = 375
    # 主点在图像中心
    # cx = width / 2.0
    # cy = height / 2.0

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
    depth_image = np.full((depth_height, depth_width), np.inf)

    # 将深度值放入对应的像素坐标 (u, v)
    for i in range(len(u)):
        if 0 <= u[i] < depth_width and 0 <= v[i] < depth_height:
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


def icp_registration(A: np.ndarray, B: np.ndarray, max_correspondence_distance: float = 0.6,
                     max_iteration: int = 2000):
    """
    使用 ICP 算法将目标点云 B 配准到源点云 A，输入点云为 numpy 格式。

    参数:
    - A: 源点云，形状为 (N, 3) 的 numpy 数组。
    - B: 目标点云，形状为 (M, 3) 的 numpy 数组。
    - max_correspondence_distance: ICP 匹配时允许的最大对应距离。
    - max_iteration: ICP 迭代的最大次数。

    返回:
    - transformation: 4x4 的刚性变换矩阵，将 B 转换到 A 坐标系下。
    - transformed_B: 经过变换后的点云 B。
    """
    # 将 numpy 数组转换为 Open3D 点云
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(A)
    target.points = o3d.utility.Vector3dVector(B)

    # 初始化变换矩阵为单位矩阵
    initial_transform = np.identity(4)

    # 执行 ICP 配准
    reg_icp = o3d.pipelines.registration.registration_icp(
        target, source, max_correspondence_distance, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    # 获取变换矩阵
    transformation = reg_icp.transformation

    # 将变换应用到目标点云 B 上
    transformed_B = np.asarray(target.transform(transformation).points).astype(int)

    return transformed_B



if __name__ == '__main__':
        # 原始数据存放位置
        # Datasets= ["qingdao","wuhu",'longhua','yuehai','lihu','yingrenshi']
        Datasets= ['yingrenshi']


        for dataset_name in Datasets:
            datas_root = "E:\dataset\\urbanbis\data\\"+str(dataset_name)+"\\"
            city_root = "E:\dataset\\urbanbis\ours\\" + str(dataset_name) + "\\"
            if not os.path.exists(city_root):
                os.mkdir(city_root)
            lists = os.listdir(datas_root)
            for file in lists:
                print("======"+file+"==========")
                # file = "2024-11-27-00-39-13"
                data_root = datas_root + file + "\images\\"
                txt_path = datas_root + file + "\\airsim_rec.txt"
                points_txt = "E:\project\\urbanbisFly\\tools\\voxel\\2voxel\\"+str(dataset_name)+".txt"
                save_remote = "/mnt/d/dataset//urbanbis//" + str(dataset_name)+"//"+file
                save_root = r"E:\dataset\\urbanbis\ours\\"+str(dataset_name)+"\\" + file + "\\"
                save_txt = save_root + "datas.txt"
                save_pkl = save_root + "datas.pkl"
                save_npy = save_root + "datas.npy"

                pfm2cam = True  # 将pfm映射到cam中
                Norm = True  # 映射到0-pc_range
                cam2pix = True  # 存放depth
                savepoints = True  # 是否生成GT标签
                meta = True  # cam2pix  pix2cam  previous_time next_time save_root
                image = True
                lidar_point = True
                mohu = True  # lidar_point 是否进行模糊查询
                pfmNorm = False
                save_pfm2cam = True
                save_points = True  # 是否存储
                save_points_quanju = False  # 是否存储
                save_depth = False
                width = 1500
                height = 800
                # 内参
                fov_degrees = 90
                # depth_width = 1500
                # depth_height = 800
                depth_width = 750
                depth_height = 400
                fx = fy = depth_width / (2 * np.tan(np.radians(fov_degrees) / 2))
                cx, cy = depth_width / 2, depth_height / 2
                intrinsic = intrinsic_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])


                # x_range = (90, 90)  # 200 units left and right
                # y_range = (0, 96)  # 200 units forward and backward
                # z_range = (90, 90)
                x_range = (96, 96)  # 200 units left and right
                y_range = (0, 128)  # 200 units forward and backward
                z_range = (96, 96)
                # 300 units downward and 0 units upward
                # pc_x_range = (60, 60)  # 200 units left and right
                # pc_y_range = (0, 64)  # 200 units forward and backward
                # pc_z_range = (60, 60)  # 300 units downward and 0 units upward
                # voxel_size = 1.5
                pc_x_range = (48, 48)  # 200 units left and right
                pc_y_range = (0, 64)  # 200 units forward and backward
                pc_z_range = (48, 48)  # 300 units downward and 0 units upward
                voxel_size = 2

                # occ_size = [120, 64, 120]
                # pc_range = [-90, -64, -90, 90, 0, 90]
                #这里就记录多一个属性
                occ_size = [96, 64, 96]
                pc_range = [-182, -96, -182, 182, 0, 182]
                txt_lists = []
                if not os.path.exists(save_root):
                    os.mkdir(save_root)
                # mesh->obj(qingdao)
                if dataset_name == "qingdao":
                    # 点云到mesh  ===qingdao====
                    # x y z
                    # 青岛
                    print("qingdao")
                    start_loacation = np.array([[-4000.0, -11531.96925, -13206.524888]]) / 100

                    AA = np.array([
                        [0.99999959, -0.00041826, 0.00080255, 43.31841575],
                        [-0.00080256, -0.00002720, 0.99999968, -200.61969388],
                        [0.00041824, 0.99999991, 0.00002754, 76.14872781],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh-》亚局部
                    BB = np.array([
                        [-0.00589500, 0.00048018, 0.99998251, 131.16643871],
                        [0.00020221, -0.99999986, 0.00048138, -115.98506379],
                        [0.99998260, 0.00020504, 0.00589490, 40.25362765],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh->点云
                    CC = np.array([
                        [0.99999959, -0.00080256, 0.00041824, -43.51125578],
                        [-0.00041826, -0.00002720, 0.99999991, -76.13606046],
                        [0.00080255, 0.99999968, 0.00002754, 200.58276691],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                #yingrenshi
                if dataset_name == "yingrenshi":
                    # 应人石
                    print("yingrenshi")
                    start_loacation = np.array([[-2563.717434, 8350.0, 550.495831]]) / 100

                # 点云到mesh
                    AA = np.array([
                    [0.99958888, 0.01316461, -0.02547092, 5.21205691],
                    [0.02550559, -0.00247239, 0.99967162, 8.71233226],
                    [-0.01309732, 0.99991029, 0.00280714, -18.35113605],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ])
                # mesh-》亚局部
                    BB = np.array([
                    [-0.00886075, -0.00344353, 0.99995481, 16.29081013],
                    [0.00401575, -0.99998613, -0.00340805, 6.95617608],
                    [0.99995268, 0.00398537, 0.00887445, -18.80632568],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ])
                # mesh->点云
                    CC = np.array([
                    [0.99958888, 0.02550559, -0.01309732, -5.67247794],
                    [0.01316461, -0.00247239, 0.99991029, 18.30241524],
                    [-0.02547092, 0.99967162, 0.00280714, -8.52520119],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ])
                #wuhu
                if dataset_name == "wuhu":
                    # 芜湖
                    print("wuhu")
                    start_loacation = np.array([[54336.521744, 8000.0, 44423.464799]]) / 100

                    # 点云到mesh
                    AA = np.array([
                        [0.99998953, -0.00447939, 0.00093300, -4.48697151],
                        [-0.00093309, -0.00001939, 0.99999956, -19.57629683],
                        [0.00447937, 0.99998997, 0.00002357, -54.91117595],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh-》亚局部
                    BB = np.array([
                        [-0.02247956, 0.01386953, 0.99965109, 49.42546163],
                        [0.00846297, -0.99986530, 0.01406281, -20.75310029],
                        [0.99971148, 0.00877614, 0.02235916, 9.14451657],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh->点云
                    CC = np.array([
                        [0.99998953, -0.00093309, 0.00447937, 4.71462550],
                        [-0.00447939, -0.00001939, 0.99998997, 54.89014660],
                        [0.00093300, 0.99999956, 0.00002357, 19.58176877],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                #longhua

                if dataset_name == "lihu":
                    print("lihu")
                    start_loacation = np.array([[60590.0,8000.0, -134760.0]]) / 100

                    # lihu_gap_location = (np.array([[15460.0, 12480.0, -134760.0]]) / 100)-start_loacation
                    # 点云到mesh
                    AA = np.array([
                        [0.99998608 ,0.00416019 ,-0.00324617 ,388.45822813],
                        [0.00324668 ,-0.00011573, 0.99999472 ,2.36797640],
                        [-0.00415979, 0.99999134, 0.00012924 ,-1379.51396795],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh-》亚局部
                    BB = np.array([
                        [-0.02480796, 0.02796198, 0.99930110, 1389.12224433],
                        [-0.03439862, -0.99904068, 0.02710073, 53.50787505],
                        [0.99910025, -0.03370226, 0.02574602, -351.91016955],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])

                    # BB = np.array([
                    #     [-0.00073447, 0.07676817, 0.99704870 ,-395.53713636],
                    #     [-0.07325160, -0.99437452, 0.07650831, 1374.67373910],
                    #     [0.99731322, -0.07297922, 0.00635373, -0.33840932],
                    #     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    # ])
                    # mesh->点云
                    CC = np.array([
                        [0.99998608, 0.00324668, -0.00415979 ,-394.19900139],
                        [0.00416019, -0.00011573, 0.99999134 ,1377.88623439],
                        [-0.00324617, 0.99999472, 0.00012924, -0.92867871],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # lihu_gap_location = apply_transformation(lihu_gap_location, BB)
                if dataset_name == "longhua":
                    # 龙华
                    start_loacation = np.array([[-26060.836348, 10000.0, -101208.9528819]]) / 100

                    AA = np.array([
                        [0.99954086, 0.00431095, 0.02999147, -163.87431504],
                        [-0.03003344, 0.01002675, 0.99949860, -4.34879178],
                        [-0.00400807, 0.99994044, -0.01015162, -70.57609906],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh-》亚局部
                    BB = np.array([
                        [-0.01586750, 0.00741042, 0.99984664, 67.12364138],
                        [-0.03790099, -0.99925833, 0.00680458, -22.90077441],
                        [0.99915551, -0.03778721, 0.01613660, 170.05149259],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh->点云
                    CC = np.array([
                        [0.99954086, -0.03003344, -0.00400807, 163.38559054],
                        [0.00431095, 0.01002675, 0.99994044, 71.32195303],
                        [0.02999147, 0.99949860, -0.01015162, 8.54498087],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                if dataset_name == "yuehai":
                    # yuehai
                    start_loacation = np.array([[-704503.735063, 5000.0, -698741.797624]]) / 100
                    # yuehai
                    # 点云到mesh
                    AA = np.array([
                        [0.99995155, -0.00277162, -0.00944514, -6876.50858895],
                        [0.00945041 ,0.00189163, 0.99995355, -8.63313346],
                        [0.00275363, 0.99999437, -0.00191773, -7058.97404732],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh-》亚局部
                    BB = np.array([
                        [-0.02949165, -0.00192790, 0.99956317, 6854.57123047],
                        [0.01250251, -0.99992062, -0.00155971, 66.17942262],
                        [0.99948683, 0.01245105, 0.02951341, 7082.03811409],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                    # mesh->点云
                    CC = np.array([
                        [0.99995155, 0.00945041 ,0.00275363, 6895.69480337],
                        [-0.00277162, 0.00189163 ,0.99999437, 7039.89155408],
                        [-0.00944514 ,0.99995355, -0.00191773, -69.85406940],
                        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                    ])
                # 相机外参
                TS = [np.array([0, 0, -10]),
                      np.array([0, 0, -10]),
                      np.array([0, 0, -10]),
                      np.array([0, 0, -10]),
                      np.array([0, 0, -10])]
                RS = [np.array([-60, 0, 0]),
                      np.array([-60, -90, 0]),
                      np.array([-60, 90, 0]),
                      np.array([-90, 0, 0]),
                      np.array([-60, -180, 0])]

                # x z y
                # 世界坐标系的启示点

                # x y z
                # start_loacation = np.array([[-4000.0, -13206.524888, -11531.96925]]) / 100
                # start_loacation = np.array([[0, 0, 0]])

                datas = read_txtdata(txt_path)

                D = np.array([
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ])

                # 真实比例是1：100  mesh:ue
                scale_factor = 100
                scale_mat = np.array([
                    [scale_factor, 0, 0, 0],
                    [0, scale_factor, 0, 0],
                    [0, 0, scale_factor, 0],
                    [0, 0, 0, 1],
                ])
                # 读取点云数据
                voxel, labels, colors = load_voxel_data_from_txt(points_txt)
                point_location = apply_transformation(start_loacation, CC)
                voxel = voxel-point_location
                # 亚局部
                # 点云-》mesh->伪局部
                filtered_voxel = apply_transformation(apply_transformation(voxel, AA), BB)
                filtered_voxel_yjb = filtered_voxel
                # np.savetxt("save.txt", filtered_voxel, delimiter=',')

                # filtered_voxel_yjb = apply_transformation(filtered_voxel, D)


                # np.savetxt("save.txt", filtered_voxel, delimiter=',')
                i = 0
                for data in tqdm(datas):
                    # 读取信息
                    # data = datas[658]
                    print(data['TimeStamp'])
                    data["x_range"] = x_range
                    data["y_range"] = y_range
                    data["z_range"] = z_range
                    data["pc_x_range"] = pc_x_range
                    data["pc_y_range"] = pc_y_range
                    data["pc_z_range"] = pc_z_range
                    data["occ_size"] = occ_size
                    data["pc_range"] = pc_range
                    position = [data["POS_X"], data["POS_Z"], data["POS_Y"]]
                    # position = [data["POS_X"],data["POS_Y"],data["POS_Z"]]
                    ori = [data['Roll'], data['Pitch'], data['Yaw']]
                    position_np = np.array([position])
                    location = start_loacation + position  # mesh坐标系
                    # points_location = apply_transformation(apply_transformation(location, BB),D)  #mesh->亚洲局部
                    # 动态外参
                    # 转成最终局部 #64 相机gap  xzy->yzx
                    T = [-position[2], position[1], (-position[0] + 64)]
                    R = [-ori[0], -ori[2], -ori[1]]
                    # 构造刚性变换矩阵 动态的转换到局部坐标系当中
                    homogeneous_matri = euler_to_homogeneous_matrix(R)
                    i_matri = inverse_rigid_transform(homogeneous_matri)  # 逆矩阵
                    filtered_voxel = apply_transformation(filtered_voxel_yjb - T, i_matri)  # 相机局部坐标系
                    # points_location = apply_transformation(points_location - T, i_matri)

                    # 制作世界标签
                    invA = np.linalg.inv(AA)
                    invB = np.linalg.inv(BB)
                    invD = np.linalg.inv(D)
                    trans_mat = translation_to_homogeneous(T)
                    air_mat = np.dot(trans_mat, np.linalg.inv(i_matri))
                    # ego2world_matri = np.dot(np.dot(np.dot(invB,D), air_mat), scale_mat)
                    # ego2world_matri = air_mat@invD@invB@invA
                    ego2world_matri = invA @ invB @ invD @ air_mat
                    ego2world_matri = scale_mat @ ego2world_matri
                    world_point = apply_transformation(filtered_voxel, ego2world_matri)
                    # np.savetxt("save.txt",world_point)
                    # world_point = apply_transformation(apply_transformation(apply_transformation(apply_transformation(filtered_voxel, air_mat),invD),invB),invA)
                    points_location = np.zeros((1, 3))
                    # 此时点云坐标系是局部的 直接从 0点开始切
                    # filtered_voxel = filtered_voxel-lihu_gap_location
                    # np.savetxt("save.txt", filtered_voxel, delimiter=',')
                    filtered_voxel, filtered_labels, filtered_colors = filter_voxels_in_box(filtered_voxel, labels, colors,
                                                                                            points_location, x_range,
                                                                                            y_range,
                                                                                            z_range, voxel_size)

                    np.savetxt("../tools/save.txt", filtered_voxel, delimiter=',')
                    # np.savetxt("save1.txt", filtered_voxel, delimiter=',')
                    # for i in range(0, 5):
                    #     matrix = create_homogeneous_matrix(TS[i], RS[i])
                    #     matrix = inverse_rigid_transform(matrix)
                    #     transformed_point_cloud = apply_transformation(filtered_voxel, matrix)
                    #     np.savetxt(str(i) + ".txt", transformed_point_cloud, delimiter=",")
                    #     depth = point_cloud_to_depth_image(transformed_point_cloud, intrinsic)
                    # np.savetxt("fortest1.txt", filtered_voxel, delimiter=",")
                    # np.savetxt("lidar_txts.txt", filtered_voxel, delimiter=',')
                    points = []  # 获取仿照lidar的点云
                    depthss = []

                    for i in range(5):
                        matrix = create_homogeneous_matrix(TS[i], RS[i])
                        matrix_in = inverse_rigid_transform(matrix)
                        R = matrix_in[0:3, 0:3]
                        T = matrix_in[0:3, 3].reshape(3, 1)
                        a = np.hstack((R, T))
                        points_3D_hom = np.hstack((filtered_voxel, np.ones((filtered_voxel.shape[0], 1))))
                        A = points_3D_hom @ (a.T)
                        depth_image = point_cloud_to_depth_image_np(A)
                        depthss.append(depth_image)
                        point = depth_to_point_cloud(depth_image)
                        # np.savetxt("save2.txt", point, delimiter=',')
                        point = HPR(point)
                        points.append(apply_transformation(point, matrix))
                    lidar_points = np.vstack(points)
                    # np.savetxt("save3.txt", lidar_points, delimiter=',')

                    if Norm:
                        # 进一步voxel化 39775

                        filtered_voxel, filtered_labels, filtered_colors = voxelize(filtered_voxel, filtered_labels,
                                                                                    filtered_colors, voxel_size=voxel_size)
                        filtered_voxel[:, 0] = (filtered_voxel[:, 0] + x_range[0] * voxel_size) / voxel_size
                        filtered_voxel[:, 1] = (filtered_voxel[:, 1] + y_range[0] * voxel_size) / voxel_size
                        filtered_voxel[:, 2] = (filtered_voxel[:, 2] + z_range[0] * voxel_size) / voxel_size

                        lidar_points = voxelize_onlypoints(lidar_points, voxel_size=voxel_size)

                        # no_norm_lidar_points  =  np.copy(lidar_points)
                        lidar_points[:, 0] = (lidar_points[:, 0] + x_range[0] * voxel_size) / voxel_size
                        lidar_points[:, 1] = (lidar_points[:, 1] + y_range[0] * voxel_size) / voxel_size
                        lidar_points[:, 2] = (lidar_points[:, 2] + z_range[0] * voxel_size) / voxel_size

                        no_norm_lidar_points = np.copy(lidar_points)
                        no_norm_lidar_points[:, 0] = (no_norm_lidar_points[:, 0] * voxel_size) - (x_range[0] * voxel_size)
                        no_norm_lidar_points[:, 1] = (no_norm_lidar_points[:, 1] * voxel_size) - (y_range[0] * voxel_size)
                        no_norm_lidar_points[:, 2] = (no_norm_lidar_points[:, 2] * voxel_size) - (z_range[0] * voxel_size)

                    if pfm2cam:  # 存放pfm转出来的深度图
                        pfm_list = [item for item in data["ImageFile"].split(";") if item.endswith(".pfm")]
                        if not os.path.exists(save_root + "pfm2cam\\"):
                            os.mkdir(save_root + "pfm2cam\\")
                        out_trans = save_root + "pfm2cam\\" + "pfm2cam_" + str(data['TimeStamp']) + ".txt"

                        # 将pfm的深度图转成局部坐标系 看看是否对的准
                        trans_points = []
                        TSs = [np.array([0, 0, -10]),
                               np.array([0, 0, -10]),
                               np.array([0, 0, -10]),
                               np.array([0, 0, -10]),
                               np.array([0, 0, -10])]
                        RSs = [np.array([-60, 0, 0]),
                              np.array([-60, -90, 0]),
                              np.array([-60, 90, 0]),
                              np.array([-90, 0, 0]),
                              np.array([-60, -180, 0])]
                        for a in range(len(pfm_list)):
                            data_name = data_root + pfm_list[a]
                            pfm_name = save_root + "pfm2cam\\" + pfm_list[a].split(".")[0] + ".txt"  # 先存储pfm转成的原始3D
                            points = pix23d(data_name)
                            transformation_matrix = create_homogeneous_matrix(TSs[a], RSs[a])
                            transformed_point_cloud = apply_transformation(points.reshape(-1, 3), transformation_matrix)
                            transformed_point_cloud = transformed_point_cloud[
                                ~np.isnan(transformed_point_cloud).any(axis=1)]
                            # 2cams
                            # np.savetxt("save"+str(a)+".txt", transformed_point_cloud, delimiter=',')

                            trans_points.append(transformed_point_cloud)
                        # np.savetxt("save.txt", np.vstack(trans_points), delimiter=',')
                        print(11)
                        if pfmNorm:
                            pfm_points = voxelize_nolabel(np.vstack(trans_points), voxel_size)
                            pfm_points[:, 0] = (pfm_points[:, 0] + x_range[0] * voxel_size) / voxel_size
                            pfm_points[:, 1] = (pfm_points[:, 1] + y_range[0] * voxel_size) / voxel_size
                            pfm_points[:, 2] = (pfm_points[:, 2] + z_range[0] * voxel_size) / voxel_size

                            # 保存pfm2cam点
                            if save_pfm2cam:
                                np.savetxt(out_trans, pfm_points,
                                           fmt='%.6f',
                                           delimiter=',')
                            # 保存pfm2cam点
                        else:
                            if save_pfm2cam:
                                np.savetxt(out_trans, np.vstack(trans_points),
                                           fmt='%.6f',
                                           delimiter=',')
                    # 存occ
                    if savepoints:
                        save_name_points = save_root + "points\\" + "pointscam_" + str(data['TimeStamp']) + ".txt"
                        if save_points:
                            if not os.path.exists(save_root + "points\\"):
                                os.mkdir(save_root + "points\\")
                            save_filtered_voxel_data_as_txt(filtered_voxel, filtered_labels, filtered_colors,
                                                            save_name_points)
                            save_name_points = save_remote + "//points//" + "pointscam_" + str(data['TimeStamp']) + ".txt"
                            data["points"] = save_name_points
                        if lidar_point:
                            set_A = set(map(tuple, filtered_voxel))
                            set_B = set(map(tuple, lidar_points))

                            # Step 2: 找到 A 和 B 的交集
                            intersection = set_A & set_B

                            # Step 3: 生成针对点云 A 的 mask
                            mask = np.array([tuple(point) in intersection for point in filtered_voxel])
                            mask_B = np.array([tuple(point) in intersection for point in lidar_points])
                            lidar_points = filtered_voxel[mask]
                            lidar_labels = filtered_labels[mask]
                            lidar_colors = filtered_colors[mask]

                            no_norm_lidar_points = np.copy(lidar_points)
                            no_norm_lidar_labels = np.copy(lidar_labels)
                            no_norm_lidar_points[:, 0] = (no_norm_lidar_points[:, 0] * voxel_size) - (
                                        x_range[0] * voxel_size)
                            no_norm_lidar_points[:, 1] = (no_norm_lidar_points[:, 1] * voxel_size) - (
                                        y_range[0] * voxel_size)
                            no_norm_lidar_points[:, 2] = (no_norm_lidar_points[:, 2] * voxel_size) - (
                                        z_range[0] * voxel_size)

                            # no_norm_lidar_points  = no_norm_lidar_points[mask_B]
                            print("利用率:")
                            print((mask.sum().item()) / (len(mask)))
                            if not os.path.exists(save_root + "lidar_point"):
                                os.mkdir(save_root + "lidar_point")
                            save_lidar_point = save_root + "lidar_point\\" + "lidar_point" + str(data['TimeStamp']) + ".txt"
                            save_lidar_point_remote = save_remote + "//lidar_point//" + "lidar_point" + str(
                                data['TimeStamp']) + ".txt"
                            data["lidar_point"] = save_lidar_point_remote
                            save_filtered_voxel_data_as_txt(lidar_points, lidar_labels, lidar_colors, save_lidar_point)
                        if save_points_quanju:
                            save_name_points_world = save_root + "points\\" + "pointscam_quanju_" + str(
                                data['TimeStamp']) + ".txt"

                            save_filtered_voxel_data_as_txt(world_point, filtered_labels, filtered_colors,
                                                            save_name_points_world)

                    if meta:
                        data["scene_token"] = file
                        if i == 0:
                            data["previous_time"] = "start_time"
                        if i != 0:
                            data["previous_time"] = datas[i - 1]["TimeStamp"]
                        if i == len(datas) - 1:
                            data["next_time"] = "end_time"
                        if i != len(datas) - 1:
                            data["next_time"] = datas[i + 1]["TimeStamp"]
                        # ---cam2liadar---
                        cam2liadar = []
                        for i in range(0, 5):
                            transformation_matrix = create_homogeneous_matrix(TS[i], RS[i])
                            cam2liadar.append(transformation_matrix)
                        if not os.path.exists(save_root + "metas\\"):
                            os.mkdir(save_root + "metas\\")
                        save_name_cam2liadar = save_root + "metas\\" + "cam2liadar_" + str(data['TimeStamp']) + ".npy"
                        save_name_cam2liadar_remote = save_remote + "//metas//" + "cam2liadar_" + str(
                            data['TimeStamp']) + ".npy"
                        np.save(save_name_cam2liadar, cam2liadar)
                        data["cam2lidar"] = save_name_cam2liadar_remote
                        # ---iamge----
                        png_list = [item for item in data["ImageFile"].split(";") if item.endswith(".png")]
                        images = []
                        save_path_names = []
                        for png_name in png_list:
                            path = data_root + png_name
                            image = cv2.imread(path)
                            if image is None:
                                print("Error: Image not found or unable to load.")

                            save_path = save_root + "imgs\\"
                            save_path_remote = save_remote + "//imgs//"
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            names = png_name.split("_")[:-1]
                            name = names[0] + "_" + names[1] + "_" + names[2] + "_" + names[3] + "_" + str(
                                data['TimeStamp']) + ".png"
                            save_path_name = save_path + name
                            save_path_name_remote = save_path_remote + name
                            # print(save_path_name)
                            save_path_names.append(save_path_name)
                            cv2.imwrite(save_path_name, image)
                            images.append(save_path_name_remote)
                        data["images_root"] = images
                    if cam2pix:  # 深度图存储
                        depths = []
                        coors = []
                        output_labels = []
                        depth_values_outputs = []

                        # 这里要换成接口的depth
                        # 将点云返回深度图
                        # pfm_list = [item for item in data["ImageFile"].split(";") if item.endswith(".pfm")]

                        # # 这个是pfm 转出来的深度图 但是因为移动问题 其实会跟实际上的有点偏差
                        # # 这里采用的是用仿真lidar来收集会准一点
                        # for pfm_path in pfm_list:
                        #     path = data_root + pfm_path
                        #     depths.append(read_pfm(path))

                        for i in range(0, 5):
                            matrix = create_homogeneous_matrix(TS[i], RS[i])
                            matrix = inverse_rigid_transform(matrix)
                            transformed_point_cloud = apply_transformation(no_norm_lidar_points, matrix)
                            # depth =  point_cloud_to_depth_image(transformed_point_cloud,intrinsic)
                            depth, coor, output_label, depth_values_output = point_cloud_to_depth_image_with_labels(
                                transformed_point_cloud, intrinsic, no_norm_lidar_labels)
                            # np.save("E:\project\\urbanbisFly\\fortest\coor"+str(i)+".npy", coor)
                            # np.save("E:\project\\urbanbisFly\\fortest\label_depth"+str(i)+".npy",depth_values_output )
                            # np.save("E:\project\\urbanbisFly\\fortest\label_seg"+str(i)+".npy", output_label)
                            # from PIL import Image
                            # image = Image.open(save_path_names[i])
                            # np.save("E:\project\\urbanbisFly\\fortest\image"+str(i)+".npy", np.array(image))
                            depths.append(depth)
                            coors.append(np.array(coor))
                            output_labels.append(np.array(output_label))
                            depth_values_outputs.append(np.array(depth_values_output))

                        # a, cam2pix_matrixs, K = Cam2pix(TS, RS, no_norm_lidar_points)
                        lidar2img = []  # 3d->uv
                        for p in range(0, 5):
                            transformation_matrix = create_homogeneous_matrix(TS[p], RS[p])
                            transformation_matrix = inverse_rigid_transform(transformation_matrix)
                            lidar2img.append(combine_matrices1(transformation_matrix, intrinsic))

                        depth_save_path = save_root + "depthAndSeg\\"
                        depth_save_path_remote = save_remote + "//depthAndSeg//"
                        if not os.path.exists(depth_save_path):
                            os.mkdir(depth_save_path)
                        depth_name = depth_save_path + str(data['TimeStamp']) + ".npy"
                        depth_name_remote = depth_save_path_remote + str(data['TimeStamp']) + ".npy"
                        if save_depth:
                            data["depth"] = depth_name_remote
                            np.save(depth_name, np.array(depths))

                        coor_name = depth_save_path + str(data['TimeStamp']) + "_coor_.pkl"
                        coor_name_remote = depth_save_path_remote + str(data['TimeStamp']) + "_coor_.pkl"
                        data["coors"] = coor_name_remote
                        with open(coor_name, "wb") as f:
                            pickle.dump(coors, f)

                        label_name = depth_save_path + str(data['TimeStamp']) + "_label_.pkl"
                        label_name_remote = depth_save_path_remote + str(data['TimeStamp']) + "_label_.pkl"
                        data["output_labels"] = label_name_remote
                        with open(label_name, "wb") as f:
                            pickle.dump(output_labels, f)

                        values_name = depth_save_path + str(data['TimeStamp']) + "_depthvalues_.pkl"
                        depthvalues_name_remote = depth_save_path_remote + str(data['TimeStamp']) + "_depthvalues_.pkl"
                        data["depth_values_outputs"] = depthvalues_name_remote
                        with open(values_name, "wb") as f:
                            pickle.dump(depth_values_outputs, f)

                        if meta:
                            cam2pix_matrixs_path = save_root + "metas\\"
                            cam2pix_matrixs_path_remote = save_remote + "//metas//"
                            if not os.path.exists(cam2pix_matrixs_path):
                                os.mkdir(cam2pix_matrixs_path)
                            save_name = cam2pix_matrixs_path + "cam2pix_" + str(data['TimeStamp']) + ".npy"
                            save_name_remote = cam2pix_matrixs_path_remote + "cam2pix_" + str(data['TimeStamp']) + ".npy"
                            np.save(save_name, lidar2img)
                            data["lidar2img"] = save_name_remote
                            save_name_k = cam2pix_matrixs_path + "intrinsic_" + str(data['TimeStamp']) + ".npy"
                            save_name_k_remote = cam2pix_matrixs_path_remote + "intrinsic_" + str(
                                data['TimeStamp']) + ".npy"  # 内参
                            np.save(save_name_k, intrinsic)
                            data["intrinsic"] = save_name_k_remote
                            save_name_ego2world = cam2pix_matrixs_path + "ego2world_" + str(data['TimeStamp']) + ".npy"
                            save_name_ego2world_remote = cam2pix_matrixs_path_remote + "ego2world_" + str(
                                data['TimeStamp']) + ".npy"  # 内参
                            np.save(save_name_ego2world, ego2world_matri)
                            data["ego2world"] = save_name_ego2world_remote

                    txt_lists.append(data)
                    # save_txt(transformed_point_cloud,save_name)
                    i = i + 1
                keys = txt_lists[0].keys()
                header = ', '.join(keys)
                import pickle

                values = [[d[key] for key in keys] for d in txt_lists]
                with open(save_pkl, 'wb') as f:
                    pickle.dump(values, f)

                values = [[str(d[key]) for key in keys] for d in txt_lists]
                np.savetxt(save_txt, values, delimiter="~", fmt="%s", comments='', header=header)
                import pickle

                with open(save_pkl, 'wb') as f:
                    pickle.dump(values, f)