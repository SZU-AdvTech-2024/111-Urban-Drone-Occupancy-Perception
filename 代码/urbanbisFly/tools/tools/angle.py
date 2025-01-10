import numpy as np
from scipy.spatial.transform import Rotation as R

"""
可以手动算角度  给出欧拉角或者四元组算
"""
def rotation_matrix_y(angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Compute cos and sin of the angle
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    # Construct the 4x4 rotation matrix around the Y-axis
    rotation_matrix = np.array([
        [cos_angle, 0., sin_angle, 0.],
        [0., 1, 0., 0.],
        [-sin_angle, 0., cos_angle, 0.],
        [0., 0., 0., 1]
    ])

    return rotation_matrix


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
    # for row in homogeneous_matrix:
    #     print(" ".join(f"{value:.8f}" for value in row))
    return  homogeneous_matrix
# Define the angle in degrees
# import numpy as np
#
# # 示例欧拉角，以弧度为单位 (roll, pitch, yaw)
# euler_angles_radians = (0.0, -1.9054073783106038, 0.0)
#
# # 将弧度转换为度数
# euler_angles_degrees = np.degrees(euler_angles_radians)
#
# print(f"Euler angles in degrees (roll, pitch, yaw): {euler_angles_degrees}")

# import numpy as np
#
# # Convert degrees to radians
# roll_rad = np.radians(0)
# pitch_rad = np.radians(5)
# yaw_rad = np.radians(0)
#
# # Euler angles (roll, pitch, yaw) in radians
# euler_angles_radians = (roll_rad, pitch_rad, yaw_rad)
#
# print(f"Euler angles in radians (roll, pitch, yaw): {euler_angles_radians}")

def euler_to_homogeneous_matrix_plus(euler_angles, translation):
    """
    Converts Euler angles and a translation vector to a 4x4 homogeneous transformation matrix.

    Parameters:
    euler_angles (tuple): A tuple of three angles (roll, pitch, yaw) in radians.
    translation (tuple): A tuple of three translation values (x, y, z).

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

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    # Insert the translation vector into the 4x4 matrix
    homogeneous_matrix = np.dot(homogeneous_matrix,translation_matrix)
    for row in homogeneous_matrix:
        print(" ".join(f"{value:.8f}" for value in row))
    # return homogeneous_matrix

if __name__ == '__main__':
    angle = -5
    # 数据原本在z轴 放到x轴  现在是负号
    # xzy R
    A = [0.00828,-2.35537,-0.00007]
    a = euler_to_homogeneous_matrix(A)
    # euler_to_homogeneous_matrix_plus([0.00828,-2.35537,-0.00007],[-15.52736,5.28277,0.62448])
    print(-10)
    # Get the rotation matrix
    transformation_matrix = rotation_matrix_y(angle)

    # Print the transformation matrix
    print("Rotation matrix around Y-axis (20 degrees):")
    for row in transformation_matrix:
        print(" ".join(f"{value:.8f}" for value in row))
