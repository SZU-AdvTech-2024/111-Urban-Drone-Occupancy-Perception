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
yaw_rate = 6
speed = 1.5
yaw_speed = 30  # 旋转速度
z_speed = 10  # 垂直速度
Colour = (0, 255, 0)
RGB = "%d %d %d" % Colour # Colour for points
depth = True
# 相机内参
Width = 750
Height = 400
CameraFOV = 90
map = "yuehai"
#----粤海---
# #play_start
# start_loacation = np.array([[-659058.125, 9414.144531, -707828.625]]) / 100
# #obj->点云坐标系
# matrix = np.array([
#     [-0.99989575, -0.01203897, -0.00797185, -6938.85853801],
#     [-0.00811055, 0.01151672, 0.99990079, 6995.48860240],
#     [-0.01194597, 0.99986120, -0.01161316, -172.69001989],
#     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
# ])
if map =='qingdao':
    # ---青岛---
    # play_start  xzy
    start_loacation = np.array([[-10216.643313, -11531.96925, -13206.524888]]) / 100
    # ========qinngdao=======
    matrix = np.array([
        [0.99999959, -0.00080256, 0.00041824, -43.51125578],
        [- 0.00041826, -0.00002720, 0.99999991, -76.13606046],
        [0.00080255, 0.99999968, 0.00002754, 200.58276691],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map =='lihu':
    # ---丽湖=== 60590.0 -134760.0  8000.0
    # ================================
    # lihu
    # start_loacation = np.array([[60590.0 ,8000.0,-134760.0]]) / 100
    start_loacation = np.array([[17260.0, 15000.0, -134760.0]]) / 100
    matrix = np.array([
        [0.99998608, 0.00324668, -0.00415979, -394.19900139],
        [0.00416019, -0.00011573, 0.99999134, 1377.88623439],
        [-0.00324617, 0.99999472, 0.00012924, -0.92867871],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    # ========lihu======

    # matrix = np.array([
    #     [0.99995155, 0.00945041, 0.00275363, 6895.69480337],
    #     [- 0.00277162, 0.00189163, 0.99999437, 7039.89155408],
    #     [- 0.00944514, 0.99995355, - 0.00191773, - 69.85406940],
    #     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    # ])

if map == 'yingrenshi':
    # --应人石==
    # start_loacation = np.array([[-2563.717434, 8350.0, 550.495831]]) / 100
    start_loacation = np.array([[-2563.717434, 8000.0, 710.4958311]]) / 100
    matrix = np.array([
        [0.99996135, -0.00563040, 0.00675256, -3.33594832],
        [-0.00670770, 0.00794928, 0.99994591, 17.04201002],
        [0.00568377, 0.99995255, -0.00791119, -6.74027688],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map == "yuehai":
    # -----粤海---
    start_loacation = np.array([[-704503.735063, 5000.0, -698741.797624]]) / 100

    # start_loacation = np.array([[-703243.735063, 10000.0, -698201.797624]]) / 100
    # ----粤海======
    matrix = np.array([
        [0.99995155, 0.00945041, 0.00275363, 6895.69480337],
        [-0.00277162, 0.00189163, 0.99999437, 7039.89155408],
        [-0.00944514, 0.99995355, -0.00191773, -69.85406940],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map == "yuehai2":
    # -----粤海---
    start_loacation = np.array([[-691173.735063, 15000.0, -710341.797624]]) / 100

    # ----粤海======
    matrix = np.array([
        [0.99995155, 0.00945041, 0.00275363, 6895.69480337],
        [-0.00277162, 0.00189163, 0.99999437, 7039.89155408],
        [-0.00944514, 0.99995355, -0.00191773, -69.85406940],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map =='wuhu':
    # -------芜湖======
    start_loacation = np.array([[-54336.521744,15000.0, 2000.0]]) / 100
    # ------wuhu----
    # matrix = np.array([
    #     [0.99998953, -0.00093309, 0.00447937, 4.71462550],
    #     [-0.00447939, -0.00001939, 0.99998997, 54.89014660],
    #     [0.00093300, 0.99999956, 0.00002357, 19.58176877],
    #     [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    # ])
    matrix = np.array([
        [0.99995271, -0.00971638, 0.00041094, 3.62667048],
        [-0.00041471, -0.00038621, 0.99999983, 57.35885843],
        [0.00971621, 0.99995272, 0.00039023, 13.86052301],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map == "longhua":
    # --------龙华------
    # start_loacation = np.array([[-26060.836348, 15000.0, -101208.9528819]]) / 100
    start_loacation = np.array([[26060.836348, 25000.0, -21208.952881]]) / 100

    # #----龙华======
    matrix = np.array([
        [0.99954086, -0.03003344, -0.00400807, 163.38559054],
        [0.00431095, 0.01002675, 0.99994044, 71.32195303],
        [0.02999147, 0.99949860, -0.01015162, 8.54498087],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
if map == 'yuehai':
    #-----粤海---
    start_loacation = np.array([[-704503.735063, 5000.0, -698741.797624]]) / 100


#obj->点云坐标系









keys_pressed = set()
number = "fortest"
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
    print(pos.x_val,pos.y_val, pos.z_val)
    # print("欧拉pos：",position)
    # 获取姿态
    orientation = client.simGetVehiclePose().orientation
    orientation_euler = airsim.to_eularian_angles(orientation)  # 转换为欧拉角
    print(orientation)
    print(orientation_euler)
    return position, orientation_euler

def depth_world(depth,filename):


    Fx = Fy = Width / (2 * math.tan(CameraFOV * math.pi / 360))
    Cx = Width / 2
    Cy = Height / 2
    #转numpy
    # depth = np.array(depth.image_data_float, dtype=np.float64)
    # depth[depth > 255] = 255
    rows, cols = depth.shape
    #2d->3d(内参)
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 800)
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
    #xzy->yzx
    pos = [pos[2],pos[1],pos[0]]
    ori = [ori[0],ori[2],ori[1]]
    #------------------scene-------------
    resps = client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False),
        airsim.ImageRequest("1", airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False),
        airsim.ImageRequest("2", airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False),
        airsim.ImageRequest("3", airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False),
        airsim.ImageRequest("4", airsim.ImageType.Scene,
                            pixels_as_float=False, compress=False)
    ])
    # for resp in resps:
    #     if resp.image_type == airsim.ImageType.Scene:
    img1d = np.fromstring(resps[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((resps[0].height, resps[0].width, 3))
    airsim.write_png(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_0.png', img_rgb)

    img1d = np.fromstring(resps[1].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((resps[1].height, resps[1].width, 3))
    airsim.write_png(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_270.png', img_rgb)

    img1d = np.fromstring(resps[1].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((resps[1].height, resps[1].width, 3))
    airsim.write_png(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_90.png', img_rgb)

    img1d = np.fromstring(resps[2].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((resps[2].height, resps[2].width, 3))
    airsim.write_png(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_Bottom.png', img_rgb)

    img1d = np.fromstring(resps[3].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape((resps[3].height, resps[3].width, 3))
    airsim.write_png(data_root + "imgs\\" + number + "\\" + nowlen + '\\Scene_180.png', img_rgb)

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
        np.save("E:\project\\urbanbisFly\photos\scene\depth\\depth.npy",depth_img_in_meters)
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


    location = start_loacation + position
    # obj->局部坐标系
    #obj坐标系换算到点云坐标系

    points = apply_transformation(location, matrix)

    file_path = r"E:\project\urbanbisFly\photos\scene\location\\"+number+".txt"
    with open(file_path, 'ab') as f:
        # f.write(header+'\n')
        # print(points)
        np.savetxt(f, points, fmt='%.8f',delimiter=',')
    for i in ori:
        pos.append(i)
    pos.append(time)
    file_path = r"E:\project\urbanbisFly\photos\scene\location\\" + number + "pos_ori.txt"
    with open(file_path, 'a') as f:
        # f.write(header+'\n')
        # print("pos",pos)
        #pos+ori+timestemp
        pos = [np.array(pos) * 1]
        np.savetxt(f, pos, fmt='%.5f', delimiter=',')
        print("over")
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
        vx = speed*20
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
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.2)
        # print("xyz",vx,vy,vz)
        # 控制旋转
        A = False
        if yaw_rate != 0:
            client.rotateByYawRateAsync(yaw_rate, 0.2)
            yaw_rate = 0
        time.sleep(0.2)
except KeyboardInterrupt:
    print("控制中断")

# 降落并解除控制
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
