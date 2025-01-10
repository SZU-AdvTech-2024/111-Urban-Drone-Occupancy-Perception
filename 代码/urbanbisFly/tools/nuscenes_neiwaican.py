# "token": 该帧数据的标识，具有唯一性
# "prev": 该帧数据上一帧数据的token，如果没有就为
# ""
# "next": 该帧数据下一帧数据的token，如果没有就为
# ""
#
# "frame_idx": 记录该帧数据是所在的序列内的第几帧，用于判断该帧数据是否为序列的首帧
# "is_key_frame": 是否为关键帧，nuscene数据集中真值数据只有2Hz，含标注信息的数据为关键帧
# "lidar_path": 该帧数据对应的lidar数据路径，注意需要是想对路径，相对于工程根目录的相对路径
#
# "sweeps": 非关键帧数据信息，原版streampetr没有用到这个信息，但是在原版本的streampetr中这个信息的作用仅仅用来判断是否为新的序列
# "cams": 记录相机的信息
# "CAM_FRONT":
# "data_path": 该帧数据这个视野的图片文件路径
# "type": 相机名称, 比如
# "CAM_FRONT"
# "timestamp": 相机时间戳
# "cam_instrnsic": 相机内参
#
# "sample_data_token": 该帧数据所在的sample的token，不参与训练
# "sensor2ego_translation": 相机外参的平移分量
# "sensor2ego_rotation": 相机外参的旋转分量, 四元数的形式[w, x, y, z]
# "ego2global_translation": 相机时间戳时刻自车系到世界系的变换的平移分量
# "ego2global_rotation": 相机时间戳时刻自车系到世界系的变换的旋转分量，四元数[w, x, y, x]
# "sensor2lidar_rotation": 相机系到激光雷达坐标系的旋转，表示一个点从相机系变换到激光雷达系的变换, 矩阵形式，因为相机和激光雷达时间戳的不一致性，
# 所以这里做了运动补偿。t时刻先从从相机系到自车系，自车系到世界系。然后T + 1
# 时刻，世界系到自车系，自车系到激光雷达系
# "sensor2lidar_translation": 相机系到激光雷达坐标系的平移
#
# "CAM_FRONT_LEFT":
# "CAM_FRONT_RIGHT":
# "CAM_BACK":
# "CAM_BACK_LEFT":
# "CAM_BACK_RIGHT":
#
# "scene_token": 该帧数据所在的场景token
# "lidar2ego_translation": 激光雷达的外参，平移分量
# "lidar2ego_rotation": 激光雷达的外参，旋转分量，四元数[w, x, y, z]
# "ego2global_translation": 激光时间戳自车系到世界系的变换，平移分量
# "ego2global_rotation": 激光时间戳自车系到世界系的变换，旋转分量
# 四元数[w, x, y, z]
#
# "timestamp": 该帧数据时间戳，使用的是激光时间戳
# "gt_boxes": 3
# D框真值, ->array
# shape = [N, 7][x, y, z, w, l, h, yaw]
# 体心世界系坐标
# "gt_names": N个object的类别 ->array
# shape = [N, ]
# "gt_velocity": N个object的横纵向速度分量 ->array
# shape = [N, 2]
# "num_lidar_pts": N个object中有多少激光点 ->array(N, )
# "num_radar_pts": N个object中有多少毫米波雷达点 ->arrayt(N, )
#
# "valid_flag": N个object是否可见, 如果num_lidar_pts > 0
# 就为可见 ->array(N, )
# "bboxes2d": ->list
# 长度为6(对应6路相机)，每个元素是数组形式，每个元素的行状为(m, 4), m表示该帧数据的所有3D框在该视野上的2D投影框有几个,
# 每一行表示为[min_x, min_y, max_x, max_y](8
# 个角点投影中的最大最新小)
# "bboxes3d_cams": ->list
# 长度为6(对应6路相机)，每个元素是数组形式，每个元素的行状为(m, 7), m对应该视野内的2D框数量，每一行都与2D投影框对应[
#     x, y, z, w, l, h, yaw], 这里的xyz是在相机坐标系，注意z轴的方向
# "label2d": ->list, 长度为6(对应6路相机)，每个元素是数组，行状为(m, )
# m表示对应相机视野内的2D投影框的类别值(原版本streampetr使用了10类，所以为0到9)
# "centers2d": ->列表，长度为6(对应6路相机)， 每个元素为数组，行状[m, 2]。记录了3D框中心点在图像上的投影像素点坐标
# "depths": ->list
# 长度为6，每个元素为数组，行状(m, )
# 记录对应物体的深度信息
# "bboxes_ignore":
# "visibilities"
#
#
