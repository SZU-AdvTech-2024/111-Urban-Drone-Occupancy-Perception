import os
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def read_point_cloud_txt(file_path):
    """
    从 TXT 文件读取点云数据，假设每行是 x, y, z, label, r, g, b 的格式
    :param file_path: 文件路径
    :return: 点云对象 (open3d.geometry.PointCloud)
    """
    data = np.loadtxt(file_path, delimiter=" ")
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


def voxelize_point_cloud(point_cloud, voxel_size=1.0):
    """
    体素化点云，按照指定的体素大小进行体素化处理
    :param point_cloud: 原始点云
    :param voxel_size: 体素大小
    :return: 体素化后的点云
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
    return voxel_grid


def visualize_point_cloud_folder_as_video(root_folder_path, frame_rate=10, enable_voxelization=False, voxel_size=1.0, visualize_images=True):
    subfolders = [f.path for f in os.scandir(root_folder_path) if f.is_dir()]

    if not subfolders:
        print("没有找到子文件夹")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    fig, axes = None, None
    if visualize_images:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for subfolder in subfolders:
        point_cloud_file = [f for f in os.listdir(subfolder) if f.endswith('.txt')]
        if not point_cloud_file:
            print(f"子文件夹 {subfolder} 中没有找到 .txt 点云文件")
            continue

        file_path = os.path.join(subfolder, point_cloud_file[0])
        print(f"显示点云文件: {file_path}")

        point_cloud = read_point_cloud_txt(file_path)

        if enable_voxelization:
            point_cloud = voxelize_point_cloud(point_cloud, voxel_size=voxel_size)

        vis.add_geometry(point_cloud)

        if visualize_images:
            image_files = [f for f in os.listdir(subfolder) if f.endswith('.jpg')][:5]
            for ax in axes:
                ax.clear()  # 清除之前的图像

            for ax, img_file in zip(axes, image_files):
                img_path = os.path.join(subfolder, img_file)
                img = Image.open(img_path)  # 使用 PIL 加载图片
                ax.imshow(img)
                ax.axis('off')

            plt.draw()
            plt.pause(1.0 / frame_rate)  # 可以调整刷新速度，减少等待时间

        vis.poll_events()
        vis.update_renderer()

        # 控制帧率
        time.sleep(1.0 / frame_rate)

        vis.clear_geometries()  # 仅在每次添加新几何体时使用

    vis.destroy_window()

if __name__ == "__main__":
    # 设置根文件夹路径
    root_folder_path = "D:\desk\\video\qingdao\\visual_dir"  # 替换为你的根文件夹路径

    # 选择是否启用体素化
    enable_voxelization = False  # True 启用体素化, False 不启用体素化

    # 选择是否可视化图片
    visualize_images = True  # True 显示图片, False 不显示图片

    # 设置体素大小
    voxel_size = 1.5  # 当启用体素化时，调整体素大小

    # 启动可视化
    visualize_point_cloud_folder_as_video(root_folder_path, frame_rate=20, enable_voxelization=enable_voxelization, voxel_size=voxel_size, visualize_images=visualize_images)
