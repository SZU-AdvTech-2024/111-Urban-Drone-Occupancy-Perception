import os

import numpy as np
from collections import Counter
from tqdm import tqdm

"""
输入的是点云 将点云voxel化之后保存下来 
可以选择不同的采样倍率
"""

def load_point_cloud_from_txt(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    labels = data[:, 6].astype(int)
    return points, labels


def voxelize_point_cloud(points, labels, voxel_size):
    # Perform voxelization
    voxel_dict = {}
    for point, label in tqdm(zip(points, labels), total=len(points), desc="Assigning points to voxels"):
        voxel_center = np.floor(point / voxel_size).astype(int)
        voxel_center_tuple = tuple(voxel_center)
        if voxel_center_tuple not in voxel_dict:
            voxel_dict[voxel_center_tuple] = []
        voxel_dict[voxel_center_tuple].append(label)

    # Determine the most frequent label in each voxel
    voxel_labels = {}
    for voxel_center, voxel_labels_list in tqdm(voxel_dict.items(), desc="Determining voxel labels"):
        most_common_label = Counter(voxel_labels_list).most_common(1)[0][0]
        voxel_labels[voxel_center] = most_common_label

    return voxel_labels


def label_to_color(label):
    # Define a color mapping for labels 0-6
    color_map = {
        0: [1, 0, 0],  # Red
        1: [0, 1, 0],  # Green
        2: [0, 0, 1],  # Blue
        3: [1, 1, 0],  # Yellow
        4: [1, 0, 1],  # Magenta
        5: [0, 1, 1],  # Cyan
        6: [1, 0.5, 0],  # Orange
    }
    return color_map.get(label, [0.5, 0.5, 0.5])  # Default to gray if label not found


def save_voxel_data_as_txt(voxel_labels, voxel_size, output_file):
    with open(output_file, 'w') as f:
        f.write("x,y,z,label,r,g,b\n")  # Write header
        for voxel_center, label in tqdm(voxel_labels.items(), desc="Saving voxel data"):
            voxel_center_coords = np.array(voxel_center) * voxel_size
            color = label_to_color(label)
            f.write(
                f"{voxel_center_coords[0]},{voxel_center_coords[1]},{voxel_center_coords[2]},{label},{color[0]},{color[1]},{color[2]}\n")
root_path = "E:\dataset\\urbanbis\点云\\"

voxel_size = 0.5
txts = os.listdir(root_path)
for i in txts:
    # Example usage
    # i = "yuehai.txt"
    input_file = root_path+str(i)
    print(i)
    txt_name = i.split(".")[0]
    output_file ="E:\project\\urbanbisFly\\tools\\voxel\\0.5voxel\\" + str(txt_name)+ str(voxel_size) + '.txt'
    # Define the voxel size
    print(output_file)
    points, labels = load_point_cloud_from_txt(input_file)
    voxel_labels = voxelize_point_cloud(points, labels, voxel_size)
    save_voxel_data_as_txt(voxel_labels, voxel_size, output_file)

    print("Voxelization complete and saved to:", output_file)
