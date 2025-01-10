import os
import numpy as np
import open3d as o3d
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import fnmatch

def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    points = data[:, :3]
    labels = data[:, 3].astype(int)
    # 翻转 Y 轴和 Z 轴
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
    colors = data[:, 4:7]
    return points_rotated, labels, colors

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
    return color_map.get(label, [0.5, 0.5, 0.5])  # Default to gray if label not found

def find_files_with_keyword(directory, keyword):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def visualize_voxel_and_images_live(txt_files, input_folder, image_folder, voxel_size, display_time=2.0):
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    # Matplotlib figure for images
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for ax in axs:
        ax.axis('off')

    # Iterate over each file
    for txt_file in txt_files:
        # Load point cloud data
        point_cloud_file = os.path.join(input_folder, txt_file)
        time_name = point_cloud_file.split("_")[-1].split(".")[0].split("t")[1]
        image_files = find_files_with_keyword(directory=image_folder, keyword=time_name)

        voxel_centers, labels, colors = load_voxel_data_from_txt(point_cloud_file)

        # Create voxel mesh
        voxel_mesh = o3d.geometry.TriangleMesh()
        for center, color in zip(voxel_centers, colors):
            box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            box.translate(center - voxel_size / 2)
            box.paint_uniform_color(color)
            voxel_mesh += box
        voxel_mesh.scale(scale_factor, center=(0, 0, 0))  # Scale around the origin (0, 0, 0)
        # Update Open3D visualizer
        vis.clear_geometries()
        vis.add_geometry(voxel_mesh)
        vis.poll_events()
        vis.update_renderer()

        # Update images in Matplotlib
        for i, (ax, img_path) in enumerate(zip(axs, image_files)):
            img = Image.open(img_path)
            ax.clear()
            ax.imshow(img)
            ax.set_title(["Front", "Left", "Right", "Bottom", "Back"][i] if i < 5 else f"View {i+1}")
            ax.axis('off')

        plt.draw()
        plt.pause(display_time)

    vis.destroy_window()

if __name__ == '__main__':
    scale_factor = 14
    data_root = "E:\dataset\\urbanbis\ours\qingdao1\\2024-09-30-16-08-49\\"
    input_folder = data_root + 'lidar_point\\'  # Point cloud TXT folder
    image_folder = data_root + '\\imgs\\'  # Image folder

    txt_files = natsorted(fnmatch.filter(os.listdir(input_folder), '*.txt'))
    voxel_size = 0.8  # Define the voxel size

    visualize_voxel_and_images_live(txt_files, input_folder, image_folder, voxel_size, display_time=0.1)
