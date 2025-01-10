import numpy as np

from tools.toUE import create_homogeneous_matrix

def inverse_rigid_transform(matrix):
    """
    Computes the inverse of a 4x4 rigid transformation matrix.

    Parameters:
    matrix (np.ndarray): A 4x4 rigid transformation matrix.

    Returns:
    np.ndarray: The inverse of the given 4x4 rigid transformation matrix.
    """
    assert matrix.shape == (4, 4), "Input matrix must be a 4x4 matrix"

    # Create an identity matrix for the inverse
    inverse_matrix = np.eye(4)

    # Transpose the 3x3 rotation part
    inverse_matrix[:3, :3] = matrix[:3, :3].T

    # Adjust the translation part
    inverse_matrix[:3, 3] = -np.dot(inverse_matrix[:3, :3], matrix[:3, 3])

    return inverse_matrix
def combine_matrices1(rigid_matrix, intrinsic_matrix):
    """
    组合4x4刚性变换矩阵和3x3内参矩阵，生成新的4x4矩阵。

    :param rigid_matrix: 4x4 刚性变换矩阵
    :param intrinsic_matrix: 3x3 内参矩阵
    :return: 4x4 组合后的矩阵
    """

    # 确保输入矩阵的形状正确
    assert rigid_matrix.shape == (4, 4), "刚性变换矩阵必须是4x4的"
    assert intrinsic_matrix.shape == (3, 3), "内参矩阵必须是3x3的"

    # 将内参矩阵扩展为4x4矩阵
    intrinsic_matrix_4x4 = np.eye(4)
    intrinsic_matrix_4x4[:3, :3] = intrinsic_matrix

    # 计算新的4x4矩阵
    combined_matrix = np.dot(intrinsic_matrix_4x4, rigid_matrix)

    return combined_matrix

homogeneous_matrix = np.array(
    [[1., 0.,  0.,  0.],
     [0.,  0., -1.,  0.],
     [0.,  1.,  0.,  0.],
     [0.,  0.,  0.,  1.]]
)
# fov_degrees = 90
# depth_width = 750
# depth_height = 400
# fx = fy = depth_width / (2 * np.tan(np.radians(fov_degrees) / 2))
# cx, cy = depth_width / 2, depth_height / 2
# intrinsic = intrinsic_matrix = np.array([
#                     [fx, 0, cx],
#                     [0, fy, cy],
#                     [0, 0, 1]
#                 ])
intrinsic = \
    np.array([[750, 0, 750],
              [0, 809, 400.],
              [0, 0., 1.]])
# lidar2cams = [
#     np.array([[1.00000000, 0.00000000, 0.00000000, 0.00000000],
#               [0.00000000, 0.76604444, -0.64278761, 0.00000000],
#               [0.00000000, 0.64278761, 0.76604444, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[-1.00000000, -0.00000000, 0.00000000, -0.00000000],
#               [0.00000000, 0.76604444, 0.64278761, 0.00000000],
#               [-0.00000000, 0.64278761, -0.76604444, -0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[0.00000000, -0.00000000, 1.00000000, 0.00000000],
#               [0.64278761, 0.76604444, -0.00000000, 0.00000000],
#               [-0.76604444, 0.64278761, 0.00000000, -0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[0.00000000, 0.00000000, -1.00000000, -0.00000000],
#               [-0.64278761, 0.76604444, -0.00000000, 0.00000000],
#               [0.76604444, 0.64278761, 0.00000000, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
#     ,
#     np.array([[1.00000000, 0.00000000, 0.00000000, 0.00000000],
#               [-0.00000000, 0.00000000, -1.00000000, -0.00000000],
#               [0.00000000, 1.00000000, 0.00000000, 0.00000000],
#               [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
# ]

# TS = [np.array([0, 0, 0]),
#       np.array([0, 0, 0]),
#       np.array([0, 0, 0]),
#       np.array([0, 0, 0]),
#       np.array([0, 0, 0])]
TS = [np.array([0, 0, -10]),
      np.array([0, 0, -10]),
      np.array([0, 0, -10]),
      np.array([0, 0, -10]),
      np.array([0, 0, -10])]
RS = [np.array([-60, 0, 0]),
      np.array([-60, -90, 0]),
      np.array([-60, 90, 0]),
      np.array([-90, 0, 0]),
      np.array([-60, -180, 0])]
transformation_matrixs = []
for i in range(0, 5):
    transformation_matrix = create_homogeneous_matrix(TS[i], RS[i])
    transformation_matrix = inverse_rigid_transform(transformation_matrix)


    transformation_matrix = combine_matrices1(transformation_matrix,intrinsic)
    # transformation_matrix =homogeneous_matrix @ homogeneous_matrix
    for row in transformation_matrix:
        print('[{:.8f}, {:.8f}, {:.8f}, {:.8f}],'.format(*row))
    print("-----------")

# 逐行打印矩阵，并格式化输出
# for row in transformation_matrixs:
#     print('{:.8f} {:.8f} {:.8f} {:.8f}'.format(*row))

# print("a2")
# for row in a2:
#     print('{:.8f} {:.8f} {:.8f} {:.8f}'.format(*row))
# a = 0
# for i in lidar2cams:
#     print("__",a,"__")
#     # A = i @ homogeneous_matrix
#     A = i
#     for row in A:
#         print('[{:.8f}, {:.8f}, {:.8f}, {:.8f}],'.format(*row))
#     a = a +1
