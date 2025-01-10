import os
import numpy as np
import open3d as o3d
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    voxel_centers = data[:, :3]
    labels = data[:, 3].astype(int)
    colors = data[:, 4:7]
    return voxel_centers, labels, colors


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


def visualize_voxel_data_as_cubes(voxel_centers, colors, voxel_size):
    mesh = o3d.geometry.TriangleMesh()
    line_set = o3d.geometry.LineSet()
    for center, color in tqdm(zip(voxel_centers, colors), desc="Creating voxels", total=len(voxel_centers)):
        box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        box.translate(center - voxel_size / 2)
        box.paint_uniform_color(color)
        mesh += box

        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        wireframe.paint_uniform_color([0, 0, 0])
        line_set += wireframe

    o3d.visualization.draw_geometries([mesh, line_set], "Voxel Visualization")


def display_images(image_paths):
    fig, axs = plt.subplots(1, len(image_paths), figsize=(20, 4))
    a = 0
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        axs[i].imshow(img)
        axs[i].axis('off')
        if a ==0:
            axs[i].set_title("front")  # 在图片下方显示文件名
        if a ==1:
            axs[i].set_title("left")  # 在图片下方显示文件名
        if a ==2:
            axs[i].set_title("right")  # 在图片下方显示文件名
        if a ==3:
            axs[i].set_title("bottom")  # 在图片下方显示文件名
        if a ==4:
            axs[i].set_title("back")  # 在图片下方显示文件名
        a = a+1
    plt.show()
def find_files_with_keyword(directory, keyword):
    matching_files = []

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file:
                # Store the full path of the matching file
                matching_files.append(os.path.join(root, file))

    return matching_files

if __name__ == '__main__':
    data_root  = "E:\dataset\\urbanbis\ours\yuehai\\2024-11-14-10-39-41\\"
    input_folder = data_root+'lidar_point\\'  # Replace with your point cloud TXT folder path
    image_folder = data_root+'\\imgs\\'  # Replace with your image folder path
    import fnmatch
    
    txt_files = natsorted(fnmatch.filter(os.listdir(input_folder), '*.txt'))
    voxel_size = 1  # Define the voxel size

    for txt_file in txt_files:
        point_cloud_file = os.path.join(input_folder, txt_file)
        time_name = point_cloud_file.split("_")[-1].split(".")[0].split("t")[1]
        image_files = find_files_with_keyword(directory=image_folder,keyword=time_name)

        # Load and visualize voxel data
        voxel_centers, labels, colors = load_voxel_data_from_txt(point_cloud_file)
        visualize_voxel_data_as_cubes(voxel_centers, colors, voxel_size)

        # Display associated images
        display_images(image_files)
