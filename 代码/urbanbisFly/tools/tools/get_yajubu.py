import numpy as np


def filter_voxels_in_box(voxel_centers, labels, colors, center_point, x_range, y_range, z_range):
    min_x, max_x = center_point[0] - x_range[0], center_point[0] + x_range[1]
    min_y, max_y = center_point[1] - y_range[0], center_point[1] + y_range[1]
    min_z, max_z = center_point[2] - z_range[0], center_point[2] + z_range[1]

    in_box = (
            ((voxel_centers[:, 0] >= min_x) & (voxel_centers[:, 0] <= max_x)) &
            ((voxel_centers[:, 1] >= min_y) & (voxel_centers[:, 1] <= max_y)) &
            ((voxel_centers[:, 2] >= min_z) & (voxel_centers[:, 2] <= max_z))
    )

    filtered_centers = voxel_centers[in_box]
    filtered_labels = labels[in_box]
    filtered_colors = colors[in_box]

    return filtered_centers, filtered_labels, filtered_colors


def load_voxel_data_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
    voxel_centers = data[:, :3]
    labels = data[:, 3].astype(int)
    colors = data[:, 4:7]
    return voxel_centers, labels, colors


def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]


x_range = (200, 200)  # 200 units left and right
y_range = (300, 300)  # 200 units forward and backward
z_range = (200, 200)  # 300 units downward and 0 units upward
#点云转mesh
# matrix = np.array([
#                         [0.99999959, -0.00041826, 0.00080255, 43.31841575],
#                         [-0.00080256, -0.00002720, 0.99999968, -200.61969388],
#                         [0.00041824, 0.99999991, 0.00002754, 76.14872781],
#                         [0.00000000, 0.00000000, 0.00000000, 1.00000000]
#                     ])
#应人石
# matrix = np.array([
#                     [0.99996135, -0.00670770, 0.00568377, 3.48844227],
#                     [-0.00563040, 0.00794928, 0.99995255, 6.58570293],
#                     [0.00675256, 0.99994591, -0.00791119, -17.07188562],
#                     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
#                 ])
#wuhu
matrix = np.array([
    [0.99998953 ,-0.00447939, 0.00093300, -4.48697151
     ],
    [-0.00093309, -0.00001939, 0.99999956, -19.57629683
     ],
    [0.00447937, 0.99998997 ,0.00002357, -54.91117595


     ],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
])
#longhua
# matrix = np.array([
#     [0.99954086, 0.00431095, 0.02999147, -163.87431504
#      ],
#     [-0.03003344, 0.01002675, 0.99949860, -4.34879178
#      ],
#     [-0.00400807, 0.99994044, -0.01015162, -70.57609906
#      ],
#     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
# ])
#yuehai

# matrix = np.array([
#     [0.99995155, -0.00277162 ,-0.00944514, -6876.50858895
#      ],
#     [0.00945041 ,0.00189163 ,0.99995355, -8.63313346
#      ],
#     [0.00275363 ,0.99999437, -0.00191773, -7058.97404732
#      ],
#     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
# ])
#lihu
# matrix = np.array([
#     [0.99998608, 0.00416019, -0.00324617, 388.45822813
#      ],
#     [0.00324668, -0.00011573, 0.99999472, 2.36797640
#      ],
#     [-0.00415979, 0.99999134, 0.00012924, -1379.51396795
#      ],
#     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
# ])
input_file = r'F:\数据集\\urbanbis数据集\点云\\Wuhu.whole.txt'  # 原始的场景
start_location = [-541.17123129,77.52589915,159.52261264
]
center_point = [0, 0, 0]
voxel_centers, labels, colors = load_voxel_data_from_txt(input_file)
voxel_centers = voxel_centers - start_location
filtered_centers, filtered_labels, filtered_colors = filter_voxels_in_box(voxel_centers, labels, colors,
                                                                          center_point, x_range,
                                                                          y_range,
                                                                          z_range)
#点云转mesh
points = apply_transformation(filtered_centers, matrix)
np.savetxt("weijubu.txt", points, delimiter=',')
