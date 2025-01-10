import numpy as np


"""
测试转到亚局部 
"""

def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

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
        next(file)
        for line in file:
            data = line.split(',')
            x, y, z = map(float, data[:3])
            points.append([x, y, z])

    return np.array(points)

def save_numpy_to_asc(points, file_path):
    """
    Saves a NumPy array as a .asc point cloud file.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) where N is the number of points.
        file_path (str): The path to the output .asc file.
    """
    with open(file_path, 'w') as file:
        for point in points:
            line = f"{point[0]} {point[1]} {point[2]}\n"
            file.write(line)
# 定义两个逆矩阵
#----yuehai----
#点云-obj
inverse_matrix_1 = np.array([
[-0.99991084, -0.01335037, 0.00026453 ,-6879.18418223],
[0.00061274, -0.02608540, 0.99965953 ,-4.25775708],
[-0.01333892, 0.99957057, 0.02609125, -7062.17947779],
[0.00000000, 0.00000000, 0.00000000, 1.00000000]
])
#obj-》亚局部
inverse_matrix_2 = np.array([
[-0.88054843, -0.00654431, -0.47391101, -9185.93761641],
[0.00484922, -0.99997673, 0.00479876, 183.71112277],
[-0.47393138, 0.00192744, 0.88055967, 3069.44346840],
[0.00000000, 0.00000000, 0.00000000, 1.00000000]
])
#----qingdao----
# 定义两个逆矩阵
#点云-obj
inverse_matrix_1 = np.array([
[0.99999959, -0.00080256, 0.00041824, -43.51125578],
[-0.00041826, -0.00002720, 0.99999991,-76.13606046],
[0.00080255, 0.99999968, 0.00002754, 200.58276691],
[0.00000000, 0.00000000, 0.00000000, 1.0000000]
])
#obj-》亚局部
inverse_matrix_2 = np.array([
[-0.88054843, -0.00654431, -0.47391101, -9185.93761641],
[0.00484922, -0.99997673, 0.00479876, 183.71112277],
[-0.47393138, 0.00192744, 0.88055967, 3069.44346840],
[0.00000000, 0.00000000, 0.00000000, 1.00000000]
])

def transform_coordinates(input_file, output_file, matrix_1, matrix_2):
    points = read_asc_to_numpy(input_file)
    transformed_coord_1 = apply_transformation(points, matrix_1)

    # 应用第二个变换矩阵
    transformed_coord_2 = apply_transformation(transformed_coord_1, matrix_2)

    save_numpy_to_asc(transformed_coord_2,output_file)
def world2obj2ego(points):
    transformed_coord_1 = apply_transformation(points, inverse_matrix_1)

    # 应用第二个变换矩阵
    transformed_coord_2 = apply_transformation(transformed_coord_1, inverse_matrix_2)
    return transformed_coord_1
    # save_numpy_to_asc(transformed_coord_2,output_file)

if __name__ == '__main__':
    # 使用函数读取和转换坐标
    input_file = 'C:\\Users\Admin\Desktop\wor\\urbanbisFly\photos\scene\points\seven\\0.txt'  # 输入ASC文件路径
    output_file = '../output.asc'  # 输出ASC文件路径
    transform_coordinates(input_file, output_file, inverse_matrix_1, inverse_matrix_2)
