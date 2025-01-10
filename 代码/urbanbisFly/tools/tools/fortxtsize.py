# import os
#
# def get_small_txt_files(folder_path, size_limit_kb=300):
#     size_limit_bytes = size_limit_kb * 1024  # 转换为字节
#     small_files = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(folder_path, filename)
#             file_size = os.path.getsize(file_path)
#             if file_size < size_limit_bytes:
#                 small_files.append(filename)
#     return small_files
#
# folder_path = 'E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-15-49-18\lidar_point\\'  # 替换为你的文件夹路径
# small_txt_files = get_small_txt_files(folder_path)
#
# print("300KB以下的TXT文件:")
# for file in small_txt_files:
#     print(file)


import os
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        # 读取第一行（表头）并跳过
        header = file.readline()

        for line in file:
            # 分割每行内容
            fields = line.strip().split('~')
            # 打印出需要的字段
            print("VehicleName:", fields[0])
            print("TimeStamp:", fields[1])
            print("Position (X, Y, Z):", fields[2], fields[3], fields[4])
            print("Quaternion (Q_W, Q_X, Q_Y, Q_Z):", fields[5], fields[6], fields[7], fields[8])
            print("ImageFile:", fields[9])
            print("Roll, Pitch, Yaw:", fields[10], fields[11], fields[12])
            print("Ranges:", fields[13:18])  # x_range, y_range, z_range, pc_x_range, pc_y_range, pc_z_range
            print("Occupancy Size:", fields[18])
            print("Point Cloud Range:", fields[19])
            print("Depth:", fields[20])
            print("Lidar to Image:", fields[21])
            print("Intrinsic:", fields[22])
            print("Ego to World:", fields[23])
            print("Previous Time:", fields[24])
            print("Next Time:", fields[25])
            print("Images Root:", fields[26])
            print("Lidar Point File:", fields[27])
            print("-" * 40)  # 分隔符


folder_path = r'E:\dataset\\urbanbis\ours\qingdao\\2024-09-29-15-49-18\\'  # 替换为你的文件夹路径

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        print(f"Reading file: {filename}")
        read_txt_file(file_path)
