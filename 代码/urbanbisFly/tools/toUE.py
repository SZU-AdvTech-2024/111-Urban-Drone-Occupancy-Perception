#位置（xyz） 旋转（）
# fc: (68, 0, -40)(0,-40,0)
# bc(-68,-10,-40)(0,-40,-180)
# bottom(0,0,-40)(0,-90,0)
# FR(-12,-80,-40)(0,-40,-90)
# FL(-12,80,-40)(0,-40,90)
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
将无人机拍的深度图转成局部坐标系
"""
Width = 750
Height = 400
CameraFOV = 90
def point_cloud_to_depth_map(point_cloud):

    # depth_map = np.zeros((Height, Width), dtype=np.float32)
    depth_map = point_cloud[:, 2].reshape(Height, Width)
    # 将点云映射到深度图中
    # for point in point_cloud:
    #     x, y, z = point
    return depth_map

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


def create_homogeneous_matrix(translation, rotation):
    """Create a homogeneous transformation matrix."""
    R = euler_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


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
        for line in file:
            data = line.split()
            x, y, z = map(float, data[:3])
            points.append([x, y, z])

    return np.array(points)
print(11)

# TS = [np.array([0, 0, -10]),
#       np.array([0, 0, -10]),
#       np.array([0, 0, -10]),
#       np.array([0, 0, -10]),
#       np.array([0, 0, -10])]
TS = [np.array([0, -10, 0]),
      np.array([0, -10, 0]),
      np.array([0, -10, 0]),
      np.array([0, -10, 0]),
      np.array([0, -10, 0])]
RS = [np.array([-60, 0, 0]),
      np.array([-60, -90, 0]),
      np.array([-60, 90, 0]),
      np.array([-90, 0, 0]),
      np.array([-60, -180, 0])]
# Define translation and rotation
# TS = [np.array([68, 0, -40]),
#       np.array([-12,80,-40]),
#       np.array([-12,-80,-40]),
#       np.array([0,0,-40]),
#       np.array([-68,-10,-40])]
#前
# RS = [np.array([0, -40, 0]),
#       np.array([0,-40,90]),
#       np.array([0,-40,-90]),
#       np.array([0,-90,0]),
#       np.array([0,-40,-180])]
# 【y z x】


def rotation_position(drone_orientation,drone_position,cloud):
    pitch, yaw, roll = drone_orientation
    rotation_matrix_roll = np.array([[1, 0, 0],
                                     [0, np.cos(roll), -np.sin(roll)],
                                     [0, np.sin(roll), np.cos(roll)]])
    rotation_matrix_pitch_yaw_y = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                                      [0, 1, 0],
                                      [np.sin(yaw), 0, np.cos(yaw)]])
    rotation_matrix_yaw_pitch_z = np.array([[np.cos(pitch), np.sin(pitch), 0],
                                    [-np.sin(pitch), np.cos(pitch), 0],
                                    [0, 0, 1]])
    # 将 roll、pitch 和 yaw 的旋转矩阵相乘得到最终的旋转矩阵
    rotation_matrix = rotation_matrix_roll.dot(rotation_matrix_pitch_yaw_y).dot(rotation_matrix_yaw_pitch_z)
    for row in rotation_matrix:
        print(" ".join(f"{value:.8f}" for value in row))
    world_point = np.dot(cloud,rotation_matrix)
    return world_point

def euler_to_homogeneous_matrix(euler_angles):
    """
    Converts Euler angles to a 4x4 homogeneous transformation matrix.

    Parameters:
    euler_angles (tuple): A tuple of three angles (roll, pitch, yaw) in radians.

    Returns:
    np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    # Create a rotation object using Euler angles
    rotation = R.from_euler('xyz', euler_angles, degrees=False)

    # Get the 3x3 rotation matrix
    rotation_matrix_3x3 = rotation.as_matrix()

    # Create a 4x4 identity matrix
    homogeneous_matrix = np.eye(4)

    # Insert the 3x3 rotation matrix into the 4x4 matrix
    homogeneous_matrix[:3, :3] = rotation_matrix_3x3
    return  homogeneous_matrix
# Example usage
def jubupoint():
    file_path_ROOT = r'E:\project\urbanbisFly\photos\scene\\tests\\'
    file_names = os.listdir(file_path_ROOT)
    points = []
    # test_file = "C:\\Users\Admin\Desktop\wor\\urbanbisFly\photos\scene\\trans0_Trans.asc"
    for i in range(len(file_names)):
        file = file_path_ROOT + file_names[i]
        point_cloud = read_asc_to_numpy(file)
        # Create transformation matrix
        #飞机的局部矩阵 将无人机的各个相机扭转到无人机的局部坐标系
        transformation_matrix = create_homogeneous_matrix(TS[i], RS[i])
        transformed_point_cloud = apply_transformation(point_cloud, transformation_matrix)

        # C = np.array(
        #     [[1, 0, 0, 0],
        #      [0, 0, 1, 0],
        #      [0, -1, 0, 0],
        #      [0, 0, 0, 1]]
        # )
        # transformed_point_cloud = apply_transformation(transformed_point_cloud, C)
        #x z轴互换  现在是加号 后面是减号 z y x T
        #这里可以理解为是 无人机的局部坐标系的转到obj坐标系中的操作（后续使用的是逆操作）
        # data = np.loadtxt(r"E:\project\urbanbisFly\photos\scene\location\\eightpos_ori.txt",delimiter=',')
        # T = data[:3]
        # R = data[3:6]
        # transformed_point_cloud = transformed_point_cloud
        # transformed_point_cloud -=T
        # homogeneous_matri = euler_to_homogeneous_matrix(R)
        # transformed_point_cloud = apply_transformation(transformed_point_cloud,homogeneous_matri)+T
        save_numpy_to_asc(transformed_point_cloud,
                          r"E:\project\urbanbisFly\photos\scene\\trans" + str(i) + "_Trans.asc")
        points.append(transformed_point_cloud)
    lidar_points = np.vstack(points)
    np.savetxt("save.txt",lidar_points,delimiter=",")
# 0.81742061 0.00000000 -0.57604127 0.00000000
# 0.00000000 1.00000000 0.00000000 0.00000000
# 0.57604127 0.00000000 0.81742061 0.00000000
# 0.00000000 0.00000000 0.00000000 1.00000000
if __name__ == '__main__':
    jubupoint()