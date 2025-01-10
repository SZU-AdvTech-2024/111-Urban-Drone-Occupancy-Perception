import os
import time
import numpy as np
import open3d as o3d


def read_point_cloud_txt(file_path):
    """
    从 TXT 文件读取点云数据，假设每行是 x, y, z, label, r, g, b 的格式
    :param file_path: 文件路径
    :return: 点云对象 (open3d.geometry.PointCloud)
    """
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    points = data[:, :3]  # 取前三列作为点的坐标
    labels = data[:, 3]  # 第四列是标签

    # 翻转 Y 轴和 Z 轴
    points[:, 1] = -points[:, 1]
    points[:, 2] = -points[:, 2]

    # 绕 X 轴旋转 45 度
    theta = np.pi / 5  # 45 度转换为弧度
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 旋转矩阵
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

    # 应用旋转
    points_rotated = points @ rotation_matrix.T  # 使用矩阵乘法进行旋转

    # 颜色映射
    colors = np.zeros((points_rotated.shape[0], 3))  # 初始化颜色数组
    for i in range(len(labels)):
        if labels[i] == 7:
            colors[i] = [1.0, 0.0, 0.0]  # 红色
        elif labels[i] == 2:
            colors[i] = [0.0, 1.0, 0.0]  # 绿色
        elif labels[i] == 1:
            colors[i] = [0.0, 0.0, 1.0]  # 蓝色
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色（其他标签）

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_rotated)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色
    return point_cloud


def voxelize_point_cloud(point_cloud, voxel_size=0.1):
    """
    体素化点云，按照指定的体素大小进行体素化处理
    :param point_cloud: 原始点云
    :param voxel_size: 体素大小
    :return: 体素化后的点云
    """
    # 体素化点云
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
    return voxel_grid


def visualize_point_cloud_folder_as_video(folder_path, frame_rate=10, enable_voxelization=False, voxel_size=1.0):
    """
    从文件夹中读取点云文件并以视频流形式可视化
    :param folder_path: 点云文件夹路径
    :param frame_rate: 视频流帧率（每秒显示的点云数）
    :param enable_voxelization: 是否启用体素化
    :param voxel_size: 体素大小，仅在启用体素化时有效
    """
    # 获取文件夹中的所有 txt 文件
    point_cloud_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

    if not point_cloud_files:
        print("没有找到 .txt 点云文件")
        return

    # 创建一个窗口用于显示点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 初始化第一个点云
    first_point_cloud = read_point_cloud_txt(os.path.join(folder_path, point_cloud_files[0]))

    # 如果启用了体素化，进行体素化处理
    if enable_voxelization:
        first_point_cloud = voxelize_point_cloud(first_point_cloud, voxel_size=voxel_size)

    vis.add_geometry(first_point_cloud)

    for file_name in point_cloud_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"显示点云文件: {file_name}")

        # 读取新点云
        point_cloud = read_point_cloud_txt(file_path)

        # 如果启用了体素化，进行体素化处理
        if enable_voxelization:
            point_cloud = voxelize_point_cloud(point_cloud, voxel_size=voxel_size)

        # 移除旧点云
        vis.remove_geometry(first_point_cloud)

        # 添加新点云
        vis.add_geometry(point_cloud)

        # 更新可视化窗口
        vis.poll_events()
        vis.update_renderer()

        # 控制帧率
        time.sleep(1.0 / frame_rate)

        # 更新第一个点云为新点云
        first_point_cloud = point_cloud

    vis.destroy_window()


if __name__ == "__main__":
    # 设置点云文件夹路径
    folder_path = "E:\dataset\\urbanbis\ours\yuehai\\2024-11-14-10-39-41\lidar_point"  # 替换为你的点云文件夹路径

    # 控制是否启用体素化模式
    enable_voxelization = True  # True 启用体素化, False 不启用体素化
    voxel_size = 0.6  # 体素大小

    visualize_point_cloud_folder_as_video(folder_path, frame_rate=1, enable_voxelization=enable_voxelization,
                                          voxel_size=voxel_size)
