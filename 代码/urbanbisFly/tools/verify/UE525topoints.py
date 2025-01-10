"""
读取rotation和translation
由UE坐标系--》Mesh坐标系--》点云坐标系
1.获取点云并转成occ
2.将Mesh坐标系-》亚局部坐标系-》相机局部坐标系-》各自的原坐标系-》3D换2D拿到对应的深度图（考虑范围？）
"""
import math

import numpy as np

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.tools.angle import euler_to_homogeneous_matrix
from tools.vis.pfmvis import pix23d

Colour = (0, 255, 0)

RGB = "%d %d %d" % Colour  # Colour for points

def to_eularian_angles(Q_W, Q_X, Q_Y, Q_Z):

    z = Q_W
    y = Q_X
    x = Q_Y
    w = Q_Z
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return pitch, roll, yaw
def quaternion_to_euler(Q_W, Q_X, Q_Y, Q_Z):
    r = R.from_quat([Q_W, Q_X, Q_Y, Q_Z])
    euler_angles = r.as_euler('xyz')  # 'xyz' 指定旋转顺序，degrees=True 表示输出角度值

    return euler_angles[0], euler_angles[1], euler_angles[2]
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

def filter_voxels_in_box(voxel_centers, labels, colors, center_point, x_range, y_range, z_range):
    min_x, max_x = center_point[0][0] - x_range[0], center_point[0][0] + x_range[1]
    min_y, max_y = center_point[0][1] - y_range[0], center_point[0][1] + y_range[1]
    min_z, max_z = center_point[0][2] - z_range[0], center_point[0][2] + z_range[1]

    in_box = (
            ((voxel_centers[:, 0] >= min_x) & (voxel_centers[:, 0] <= max_x)) &
            ((voxel_centers[:, 1] >= min_y) & (voxel_centers[:, 1] <= max_y)) &
            ((voxel_centers[:, 2] >= min_z) & (voxel_centers[:, 2] <= max_z))
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

def save_filtered_voxel_data_as_txt(voxel_centers, labels, colors, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z,label,r,g,b\n")
        for center, label, color in zip(voxel_centers, labels, colors):
            f.write(f"{center[0]},{center[1]},{center[2]},{label},{color[0]},{color[1]},{color[2]}\n")
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


if __name__ == '__main__':
    points_txt = "E:\project\\urbanbisFly\\tools\\voxel\qingdao1.txt"
    txt_path = "E:\project\\urbanbisFly\\tools\\tools\\a.txt"
    txt_save = "E:\project\\urbanbisFly\\tools\out_asc\\"
    data_root = "C:\\Users\Admin\Documents\AirSim\\2024-08-28-10-37-03\\"
    #第一次粗提取 开大点
    x_range = (500, 500)  # 200 units left and right
    y_range = (500, 500)  # 200 units forward and backward
    z_range = (500, 500)  # 300 units downward and 0 units upward
    voxel_size = 2
    #mesh->obj(qingdao)
    # 点云到mesh
    A = np.array([
        [0.99999959, -0.00041826, 0.00080255, 43.31841575],
        [-0.00080256, -0.00002720, 0.99999968, -200.61969388],
        [0.00041824, 0.99999991, 0.00002754, 76.14872781],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    # mesh-》亚局部
    B = np.array([
        [-0.00589500, 0.00048018, 0.99998251, 131.16643871],
        [0.00020221, -0.99999986, 0.00048138, -115.98506379],
        [0.99998260, 0.00020504, 0.00589490, 40.25362765],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    #mesh->点云
    C = np.array([
        [0.99999959, -0.00080256, 0.00041824, -43.51125578],
        [-0.00041826, -0.00002720, 0.99999991, -76.13606046],
        [0.00080255, 0.99999968, 0.00002754, 200.58276691],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    #相机内参
    TS = [np.array([0, 0, 0]),
          np.array([0, 0, 0]),
          np.array([0, 0, 0]),
          np.array([0, 0, 0]),
          np.array([0, 0, 0])]
    RS = [np.array([-40, 0, 0]),
          np.array([-40, -90, 0]),
          np.array([-40, 90, 0]),
          np.array([-90, 0, 0]),
          np.array([-40, -180, 0])]

    # x z y
    #世界坐标系的启示点
    start_loacation = np.array([[-4000.0, -11531.96925, -13206.524888]]) / 100
    # x y z
    # start_loacation = np.array([[-4000.0, -13206.524888, -11531.96925]]) / 100
    start_loacation = np.array([[0, 0, 0]])

    datas = read_txtdata(txt_path)

    voxel, labels, colors = load_voxel_data_from_txt(points_txt)
    i = 0
    output_file = "E:\project\\urbanbisFly\\tools\\out\\"
    for data in datas:

        position = [data["POS_X"], data["POS_Z"], data["POS_Y"]]
        # position = [data["POS_X"],data["POS_Y"],data["POS_Z"]]

        ori = [data['Roll'], data['Pitch'], data['Yaw']]
        position_np = np.array([position])
        location = start_loacation + position
        points_location = apply_transformation(location, C)  #点云坐标系


        # R = [-ori[1],ori[2],ori[0]]
        # T = [position[1], position[2], position[0]]
        # R = ori
        filtered_voxel, filtered_labels, filtered_colors = filter_voxels_in_box(voxel, labels, colors,
                                                                                points_location, x_range,
                                                                                y_range,
                                                                                z_range)
        filtered_voxel = apply_transformation(apply_transformation(filtered_voxel, A), B)

        D = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
        filtered_voxel = apply_transformation(filtered_voxel, D)
        #转成最终局部 #64 相机gap  xzy->yzx
        T = [-position[2], position[1], -position[0]+64]
        R = [-ori[0], -ori[2], ori[1]]
        # R = [ori[1],ori[2],ori[0]]
        # R = [ori[2],ori[1],ori[0]]
        homogeneous_matri = euler_to_homogeneous_matrix(R)
        i_matri = inverse_rigid_transform(homogeneous_matri)
        filtered_voxel = apply_transformation(filtered_voxel - T, i_matri)

        # xx_range = (-36, 36)
        # yy_range = (-36, 36)
        # zz_range = (-52, 4)
        xx_range = (-100, 100)
        yy_range = (-100, 100)
        zz_range = (-100, 100)

        mask = (
                (filtered_voxel[:, 0] >= xx_range[0] * voxel_size) & (
                filtered_voxel[:, 0] <= xx_range[1] * voxel_size) &
                (filtered_voxel[:, 1] >= yy_range[0] * voxel_size) & (
                        filtered_voxel[:, 1] <= yy_range[1] * voxel_size) &
                (filtered_voxel[:, 2] >= zz_range[0] * voxel_size) & (
                        filtered_voxel[:, 2] <= zz_range[1] * voxel_size)
        )

        filtered_voxel = filtered_voxel[mask]

        filtered_labels = filtered_labels[mask]
        filtered_colors = filtered_colors[mask]

        # 进一步voxel化 映射到0-pcrange中
        # filtered_voxel, filtered_labels, filtered_colors = voxelize(filtered_voxel, filtered_labels,
        #                                                               filtered_colors, voxel_size=2)
        # filtered_voxel[:, 0] = np.floor((filtered_voxel[:, 0] - xx_range[0] * voxel_size) / 2)
        # filtered_voxel[:, 1] = np.floor((filtered_voxel[:, 1] - yy_range[0] * voxel_size) / 2)
        # filtered_voxel[:, 2] = np.floor((filtered_voxel[:, 2] - zz_range[0] * voxel_size) / 2)

        output_file_txt = output_file + str(i) + ".txt"
        save_filtered_voxel_data_as_txt(filtered_voxel, filtered_labels, filtered_colors, output_file_txt)
        print(i)
        pfm_list = [item for item in data["ImageFile"].split(";") if item.endswith(".pfm")]
        out_trans = "E:\project\\urbanbisFly\\tools\out_trans\\"

        trans_points = []
        for a in range(len(pfm_list)):
            data_name = data_root + pfm_list[a]
            save_name = txt_save + pfm_list[a].split(".")[0] + ".txt"
            points = pix23d(data_name, save_name)
            points = read_asc_to_numpy(save_name)
            save_name = out_trans + pfm_list[a].split(".")[0] + ".txt"
            transformation_matrix = create_homogeneous_matrix(TS[a], RS[a])
            transformed_point_cloud = apply_transformation(points, transformation_matrix)
            trans_points.append(transformed_point_cloud)
        np.savetxt("E:\project\\urbanbisFly\\tools\out_trans\\" + str(i) + ".txt", np.vstack(trans_points), fmt='%.6f',
                   delimiter=',')

        # save_txt(transformed_point_cloud,save_name)
        i = i + 1
