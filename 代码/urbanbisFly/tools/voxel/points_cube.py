import os.path

import numpy as np

from tools.tools.angle import euler_to_homogeneous_matrix

"""
点云-》mesh-》亚局部-》最终局部    voxel切割
"""
def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    voxel_centers = data[:, :3]
    labels = data[:, 3].astype(int)
    colors = data[:, 4:7]
    return voxel_centers, labels, colors


def filter_voxels_in_box(voxel_centers, labels, colors, center_point, x_range, y_range, z_range):
    min_x, max_x = center_point[0] - x_range[0], center_point[0] + x_range[1]
    min_y, max_y = center_point[1] - y_range[0], center_point[1] + y_range[1]
    min_z, max_z = center_point[2] - z_range[0], center_point[2] + z_range[1]

    in_box = (
            ((voxel_centers[:, 0] >= min_x) & (voxel_centers[:, 0] <= max_x)) &
            ((voxel_centers[:, 1] >= min_y) & (voxel_centers[:, 1] <= max_y)) &
            ((voxel_centers[:, 2] >= min_z) & (voxel_centers[:, 2] <= max_z))
    )

    filtered_centers = voxel_centers[in_box]
    filtered_labels = labels[in_box]
    filtered_colors = colors[in_box]

    return filtered_centers, filtered_labels, filtered_colors


def save_filtered_voxel_data_as_txt(voxel_centers, labels, colors, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z,label,r,g,b\n")
        for center, label, color in zip(voxel_centers, labels, colors):
            f.write(f"{center[0]},{center[1]},{center[2]},{label},{color[0]},{color[1]},{color[2]}\n")


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


if __name__ == '__main__':
    """
    给出对应的裁剪区域
    """
    # 用于初次切割的 不然每次都载这么大的图会很慢
    x_range = (500, 500)  # 200 units left and right
    y_range = (500, 500)  # 200 units forward and backward
    z_range = (500, 500)  # 300 units downward and 0 units upward
    input_file = 'voxelized_point_cloud2.txt'  # 原始的场景
    # number = "one"
    numbers =  os.listdir(r"E:\project\urbanbisFly\photos\scene\imgs")
    for number  in numbers:
        number = str(number)
        center_file = r"E:\project\urbanbisFly\photos\scene\location\\" + number + ".txt"
        pors_ori_file = r"E:\project\urbanbisFly\photos\scene\location\\" + number + "pos_ori.txt"
        oriandpos = np.loadtxt(pors_ori_file, delimiter=",")
        voxel_size = 2
        save_txt = True
        with open(center_file, 'r') as file:
            lines = file.readlines()
            i = 0
            for line in lines:  # Skip the header line
                output_file = r'E:\project\urbanbisFly\photos\scene\points\\test\\' + number + '\\'
                cx, cy, cz = map(float, line.strip().split(','))
                center_point = [cx, cy, cz]
                if not os.path.exists(output_file):
                    os.mkdir(output_file)
                # 读取点云
                voxel_centers, labels, colors = load_voxel_data_from_txt(input_file)
                # 仿射变换
                filtered_centers, filtered_labels, filtered_colors = filter_voxels_in_box(voxel_centers, labels, colors,
                                                                                          center_point, x_range,
                                                                                          y_range,
                                                                                          z_range)
                # 点云到mesh
                A = np.array([
                    [-0.99991084, -0.01335037, 0.00026453, -6879.18418223],
                    [0.00061274, -0.02608540, 0.99965953, -4.25775708],
                    [-0.01333892, 0.99957057, 0.02609125, -7062.17947779],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ])
                # mesh-》亚局部
                B = np.array([
                    [0.00747532, 0.00355394, 0.99996574, 7128.15722763],
                    [-0.00404680, -0.99998539, 0.00358426, 90.62354574],
                    [0.99996387, -0.00407345, -0.00746083, 6538.29701216],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                ])
                filtered_centers = apply_transformation(apply_transformation(filtered_centers, A), B)
                # 转成最终局部
                data = oriandpos[i]
                T = data[:3]
                R = data[3:6]
                # transformed_point_cloud = transformed_point_cloud
                R = R.tolist()
                homogeneous_matri = euler_to_homogeneous_matrix(R)
                i_matri = inverse_rigid_transform(homogeneous_matri)
                filtered_centers = apply_transformation(filtered_centers - T, i_matri)
                # 这个是转成类似与ego的矩阵
                C = np.array(
                    [[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]]
                )
                filtered_centers = apply_transformation(filtered_centers, C)
                # x_range = (150, 150)  # 200 units left and right
                # y_range = (50, 150)  # 200 units forward and backward
                # z_range = (150, 150)  # 300 units downward and 0 units upward
                # filtered_centers, filtered_labels, filtered_colors = filter_voxels_in_box(filtered_centers, filtered_labels, filtered_colors,
                #                                                                           [0,0,0], x_range, y_range,
                #                                                                           z_range)
                # 获取voxel尺寸

                xx_range = (-36, 36)
                yy_range = (-36, 36)
                zz_range = (-52, 4)
                mask = (
                        (filtered_centers[:, 0] >= xx_range[0] * voxel_size) & (
                            filtered_centers[:, 0] <= xx_range[1] * voxel_size) &
                        (filtered_centers[:, 1] >= yy_range[0] * voxel_size) & (
                                    filtered_centers[:, 1] <= yy_range[1] * voxel_size) &
                        (filtered_centers[:, 2] >= zz_range[0] * voxel_size) & (
                                    filtered_centers[:, 2] <= zz_range[1] * voxel_size)
                )

                filtered_centers = filtered_centers[mask]
                filtered_labels = filtered_labels[mask]
                filtered_colors = filtered_colors[mask]
                # 进一步voxel化
                filtered_centers, filtered_labels, filtered_colors = voxelize(filtered_centers, filtered_labels,
                                                                              filtered_colors, voxel_size=2)
                filtered_centers[:,0] = np.floor((filtered_centers[:,0]-xx_range[0]*voxel_size)/2)
                filtered_centers[:,1] = np.floor((filtered_centers[:,1] - yy_range[0]*voxel_size) / 2)
                filtered_centers[:,2] = np.floor((filtered_centers[:,2] - zz_range[0]*voxel_size) / 2)

                output_file = output_file + str(i) + ".npy"
                data = np.hstack((filtered_centers, filtered_labels.reshape(-1, 1)))
                np.save(output_file, data)
                if save_txt:
                    output_file_txt = output_file + str(i) + ".txt"
                    save_filtered_voxel_data_as_txt(filtered_centers, filtered_labels, filtered_colors, output_file_txt)
                i += 1
                print("Filtered voxel data saved to:", output_file)
