import os
import numpy as np
import open3d as o3d
from natsort import natsorted
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 读取点云数据的函数
def label_to_color(label):
    color_map = {
        1: [1, 0, 0],  # Red
        2: [0, 1, 0],  # Green
        3: [0, 0, 1],  # Blue
        4: [1, 1, 0],  # Yellow
        5: [1, 0, 1],  # Magenta
        6: [0, 1, 1],  # Cyan
        7: [1, 0.5, 0],  # Orange
    }
    # 如果标签不在映射中，则返回灰色
    return color_map.get(label, [0.5, 0.5, 0.5])


# 读取点云数据的函数
def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=' ')
    points = data[:, :3]  # 点云中心坐标
    labels = data[:, 3].astype(int)  # 标签
    # 根据标签给点云分配颜色
    colors = np.array([label_to_color(label) for label in labels])
    points[:, 1] = -points[:, 1]
    points[:, 2] = -points[:, 2]
    # 绕 X 轴旋转 45 度
    theta = np.pi / 3  # 45 度转换为弧度
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
    return points_rotated, labels, colors


# 可视化点云数据的函数
def visualize_voxel_data(voxel_centers, colors, voxel_size):
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将点云中心坐标转换为 numpy 数组
    point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)

    # 将颜色设置为相应的值
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 创建 Open3D 可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    # 添加点云到可视化器
    vis.add_geometry(point_cloud)

    # 调整视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)

    return vis, point_cloud


# 遍历文件夹中的子文件夹，查找并加载 'pred.txt'
def find_subfolders_with_pred_txt(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        if 'pred.txt' in files:
            subfolders.append(root)  # 添加包含 'pred.txt' 文件的子文件夹
    return subfolders


# 主函数，执行点云加载与可视化
def main():
    data_root = "D:\desk\\video\qingdao\\visual_dir\\"

    vis, point_cloud = None, None

    subfolders = find_subfolders_with_pred_txt(data_root)  # 获取所有包含 'pred.txt' 的子文件夹

    voxel_size = 2# 定义体素大小

    # 遍历每个子文件夹中的 'pred.txt'
    # 遍历每个子文件夹中的 'pred.txt'
    for subfolder in subfolders:
        point_cloud_file = os.path.join(subfolder, 'pred.txt')

        # 加载点云数据
        voxel_centers, labels, colors = load_voxel_data_from_txt(point_cloud_file)

        # 如果是第一次创建可视化窗口，初始化
        if vis is None:
            vis, point_cloud = visualize_voxel_data(voxel_centers, colors, voxel_size)
        else:
            # 更新点云数据
            point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # 刷新可视化窗口
            vis.update_geometry(point_cloud)
        vis.get_render_option().point_size = voxel_size   # 增大显示点的大小

        # 每次展示新的点云数据后，稍作停留以查看效果
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.5)  # 这里可以根据需要调整帧间隔时间（单位：秒）

    # 结束可视化
    vis.destroy_window()


if __name__ == '__main__':
    main()
