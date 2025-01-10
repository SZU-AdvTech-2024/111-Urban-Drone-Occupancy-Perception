import os
import threading

import airsim
import time
import math
import cv2
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from datetime import datetime
# 连接AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
data_root = r"E:\project\urbanbisFly\photos\scene\\"
# 起飞
client.takeoffAsync().join()
"""
无人机数据采集 可以保存深度图 原图 还有转换后的坐标
"""
# 初始速度和控制变量
vx, vy, vz = 0, 0, 0
yaw_rate = 0
speed = 6
yaw_speed = 30  # 旋转速度
z_speed = 15  # 垂直速度
Colour = (0, 255, 0)
RGB = "%d %d %d" % Colour # Colour for points
depth = True


keys_pressed = set()
number = "seven"
def savepointcloud(image, filename):

    f = open(filename, "w+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if pt[0]!=0:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], RGB))
    f.close()

def get_drone_position_and_orientation():
    # 获取位置
    pos = client.simGetVehiclePose().position
    position = np.array([ pos.x_val,pos.z_val, pos.y_val])
    # print("欧拉pos：",position)
    # 获取姿态
    orientation = client.simGetVehiclePose().orientation
    orientation_euler = airsim.to_eularian_angles(orientation)  # 转换为欧拉角

    return position, orientation_euler

def depth_world(depth,filename):
    #相机内参
    Width = 1500
    Height = 800
    CameraFOV = 90

    Fx = Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
    Cx = Width / 2
    Cy = Height / 2
    #转numpy
    # depth = np.array(depth.image_data_float, dtype=np.float64)
    depth[depth > 255] = 255
    rows, cols = depth.shape
    #2d->3d(内参)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    # z = 100 * np.where(valid, depth / 256.0, np.nan)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - Cx) / Fx, 0)
    y = np.where(valid, z * (r - Cy) / Fy, 0)
    point = np.dstack((x, y, z))

    #相机外参 平移 绝对值
    savepointcloud(point,filename)



def take_photo():
    if not os.path.exists(data_root + "imgs\\" + number):
        os.mkdir(data_root + "imgs\\" + number)
    if depth:
        if not os.path.exists(data_root + "depth\\" + number):
            os.mkdir(data_root + "depth\\" + number)

    number_list = os.listdir(data_root+"imgs\\"+number+"\\")
    nowlen = str(len(number_list))

    #-----------------创建文件夹---------
    os.mkdir(data_root+"imgs\\"+number+"\\"+nowlen)
    if depth:
        os.mkdir(data_root + "depth\\" + number + "\\" + nowlen)
    pos,ori = get_drone_position_and_orientation()

    pos = [pos[2],pos[1],pos[0]]
    ori = [ori[0],ori[2],ori[1]]
    #------------------scene-------------
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene,
                                                         pixels_as_float=False, compress=True),
                                     airsim.ImageRequest(1, airsim.ImageType.Scene,
                                                         pixels_as_float=False, compress=True),
                                     airsim.ImageRequest(2, airsim.ImageType.Scene,
                                                         pixels_as_float=False, compress=True),
                                     airsim.ImageRequest(3, airsim.ImageType.Scene,
                                                         pixels_as_float=False, compress=True),
                                     airsim.ImageRequest(4, airsim.ImageType.Scene,
                                                         pixels_as_float=False, compress=True)
                                     ])
    f = open(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_0.png', 'wb')

    f.write(responses[0].image_data_uint8)

    f = open(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_270.png', 'wb')
    f.write(responses[1].image_data_uint8)

    f = open(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_90.png', 'wb')
    f.write(responses[2].image_data_uint8)

    f = open(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_Bottom.png', 'wb')
    f.write(responses[3].image_data_uint8)

    f = open(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_180.png', 'wb')
    f.write(responses[4].image_data_uint8)
    f.close()

    #------------------------depth-----------------

    if depth:
        responses = client.simGetImages(
            [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
             airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
             airsim.ImageRequest(2, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
             airsim.ImageRequest(3, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
             airsim.ImageRequest(4, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
             ])

        # print("pos")
        # print(pos)
        # print("ori")
        # print(ori)
        depth_img_in_meters = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                            responses[0].height)
        depth_img_in_meters = depth_img_in_meters.reshape(responses[0].height, responses[0].width)

        # center front
        depth_world(depth_img_in_meters, data_root+"\\tests\\test0.asc")
        cv2.imwrite(data_root + "depth\\" + nowlen + '\\\\depth_0.png', depth_img_in_meters)
        # center left
        depth_img_in_meters = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width,
                                                            responses[1].height)
        depth_img_in_meters = depth_img_in_meters.reshape(responses[1].height, responses[1].width)
        depth_world(depth_img_in_meters, data_root+"\\tests\\test1.asc")
        cv2.imwrite(data_root + "depth\\" + nowlen + '\\\\depth_270.png', depth_img_in_meters)
        # right
        depth_img_in_meters = airsim.list_to_2d_float_array(responses[2].image_data_float, responses[2].width,
                                                            responses[2].height)
        depth_img_in_meters = depth_img_in_meters.reshape(responses[2].height, responses[2].width)
        depth_world(depth_img_in_meters, data_root+"\\tests\\test2.asc")
        cv2.imwrite(data_root + "depth\\" + nowlen + '\\\\depth_90.png', depth_img_in_meters)
        # bottom
        depth_img_in_meters = airsim.list_to_2d_float_array(responses[3].image_data_float, responses[3].width,
                                                            responses[3].height)
        depth_img_in_meters = depth_img_in_meters.reshape(responses[3].height, responses[3].width)
        depth_world(depth_img_in_meters, data_root+"\\tests\\test3.asc")
        cv2.imwrite(data_root + "depth\\" + nowlen + '\\\\depth_Bottom.png', depth_img_in_meters)
        # back
        depth_img_in_meters = airsim.list_to_2d_float_array(responses[4].image_data_float, responses[4].width,
                                                            responses[4].height)
        depth_img_in_meters = depth_img_in_meters.reshape(responses[4].height, responses[4].width)
        depth_world(depth_img_in_meters, data_root+"\\tests\\test4.asc")
        cv2.imwrite(data_root + "depth\\" + nowlen + '\\\\depth_180.png', depth_img_in_meters)
        print("save_depth")
    return pos,ori
def on_press(key):
    global vx, vy, vz, yaw_rate
    try:
        keys_pressed.add(key.char)
        update_movement()
    except AttributeError:
        pass

def on_release(key):
    global vx, vy, vz, yaw_rate
    try:
        keys_pressed.discard(key.char)
        update_movement()
    except AttributeError:
        pass

def apply_transformation(points, transformation_matrix):
    """Apply homogeneous transformation to a set of 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]
def k_action():
    position, orientation_euler = get_drone_position_and_orientation()
    pos,ori = take_photo()  # 拍照
    time = int(datetime.now().timestamp() * 1_000_000)
    print("Current Timestamp (seconds):", time)
    # UE坐标系换算到obj坐标系
    start_loacation = np.array([[-659058.125, 9414.144531, -707828.625]]) / 100

    location = start_loacation + position
    # obj->mesh 的刚性变换矩阵
    #obj坐标系换算到点云坐标系
    matrix = np.array([
        [-0.99989575 ,-0.01203897 ,-0.00797185 ,-6938.85853801],
        [-0.00811055 ,0.01151672 ,0.99990079 ,6995.48860240],
        [-0.01194597 ,0.99986120 ,-0.01161316 ,-172.69001989],
        [0.00000000 ,0.00000000 ,0.00000000 ,1.00000000]
    ])
    points = apply_transformation(location, matrix)

    file_path = r"E:\project\urbanbisFly\photos\scene\location\\"+number+".txt"
    with open(file_path, 'ab') as f:
        # f.write(header+'\n')
        print(points)
        np.savetxt(f, points, fmt='%.8f',delimiter=',')
    for i in ori:
        pos.append(i)
    pos.append(time)
    file_path = r"E:\project\urbanbisFly\photos\scene\location\\" + number + "pos_ori.txt"
    with open(file_path, 'a') as f:
        # f.write(header+'\n')
        print("pos",pos)
        #pos+ori+timestemp
        pos = [np.array(pos) * 1]
        np.savetxt(f, pos, fmt='%.5f', delimiter=',')
def update_movement():
    global vx, vy, vz, yaw_rate
    global A
    A = False
    vx, vy, vz, yaw_rate = 0, 0, 0, 0
    # print(keys_pressed)
    if 'w' in keys_pressed:
        vx = speed
    if 's' in keys_pressed:
        vx = -speed
    if 'q' in keys_pressed:
        yaw_rate = -yaw_speed
    if 'e' in keys_pressed:
        yaw_rate = yaw_speed
    if 'o' in keys_pressed:
        vz = -z_speed  # 向上飞
    if 'p' in keys_pressed:
        vz = z_speed  # 向下飞
    if 'k' in keys_pressed:
        threading.Thread(target=k_action).start()
    # 加速
    if 'l' in keys_pressed:
        vx = speed*5
    # if 'j' in keys_pressed:
    #     for i in range(10):
    #         client.moveByVelocityAsync(2, 0, 0, 5)
    #         k_action()
    #         time.sleep(1)

# 设置键盘监听器
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

try:
    while True:
        # 移动无人机（本地坐标系）
        # pr  int(yaw_rate)
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.3)
        # print("xyz",vx,vy,vz)
        # 控制旋转
        A = False
        if yaw_rate != 0:
            client.rotateByYawRateAsync(yaw_rate, 0.3)
            yaw_rate = 0
        time.sleep(0.3)
except KeyboardInterrupt:
    print("控制中断")

# 降落并解除控制
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
