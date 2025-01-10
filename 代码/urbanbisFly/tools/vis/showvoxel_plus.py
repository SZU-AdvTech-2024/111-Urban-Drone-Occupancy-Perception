import os
import numpy as np
import open3d as o3d
from natsort import natsorted
from tqdm import tqdm

"""
展示采集的occ，自动判断是否有颜色信息
"""


def load_voxel_data_from_txt(file_path):
    # 读取数据，检查列数以判断是否有颜色信息
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    voxel_centers = data[:, :3]  # 前三列是坐标
    labels = data[:, 3].astype(int)  # 第四列是标签
    if data.shape[1] > 4:  # 如果列数超过4，则包含颜色信息
        colors = data[:, 4:7]  # 提取颜色信息
    else:
        colors = None  # 没有颜色信息
    return voxel_centers, labels, colors


def label_to_color(label):
    # 标签对应颜色映射
    color_map = {
        1: [1, 0, 0],  # Red (Terrain)
        2: [0, 1, 0],  # Green (Vegetation)
        3: [0, 0, 1],  # Blue (Water)
        4: [1, 1, 0],  # Yellow (Bridge)
        5: [1, 0, 1],  # Magenta (Vehicle)
        6: [0, 1, 1],  # Cyan (Boat)
        7: [1, 0.5, 0],  # Orange (Building)
    }
    return color_map.get(label, [0.5, 0.5, 0.5])  # 默认灰色


def visualize_voxel_data_as_cubes(voxel_centers, labels, voxel_size, colors=None):
    # 创建体素网格和线框
    mesh = o3d.geometry.TriangleMesh()
    line_set = o3d.geometry.LineSet()

    for i, (center, label) in tqdm(enumerate(zip(voxel_centers, labels)), desc="Creating voxels",
                                   total=len(voxel_centers)):
        if colors is not None:
            # 如果有颜色信息，则使用给定的颜色
            color = colors[i]
        else:
            # 如果没有颜色信息，则使用标签映射的颜色
            color = label_to_color(label)

        # 创建体素立方体
        box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        box.translate(center - voxel_size / 2)
        box.paint_uniform_color(color)
        mesh += box

        # 创建线框
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        wireframe.paint_uniform_color([0, 0, 0])  # 线框颜色为黑色
        line_set += wireframe

    # 使用Open3D进行可视化
    o3d.visualization.draw_geometries([mesh, line_set], "Voxel Visualization")


if __name__ == '__main__':
    # 示例使用
    input_file = 'E:\\project\\urbanbisFly\\screen\\'  # 你的TXT文件路径
    import fnmatch

    lists = natsorted(os.listdir(input_file))
    lists = fnmatch.filter(lists, '*.txt')
    voxel_size = 1.5  # 定义体素大小

    for file in lists:
        file_path = os.path.join(input_file, file)
        voxel_centers, labels, colors = load_voxel_data_from_txt(file_path)
        visualize_voxel_data_as_cubes(voxel_centers, labels, voxel_size, colors)
