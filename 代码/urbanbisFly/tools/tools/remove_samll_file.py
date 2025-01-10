import os

# 指定文件夹路径
folder_path = r"E:\dataset\urbanbis\ours\qingdao\2024-09-29-14-21-30\lidar_point"
small_files = []  # 用于存储小于600KB的文件名列表

# 600KB的大小限制（600KB = 600 * 1024字节）
size_limit = 550 * 1024

# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        # 获取文件大小（字节）
        file_size = os.path.getsize(file_path)
        # 检查文件大小是否小于600KB
        if file_size < size_limit:
            small_files.append(file_name)  # 只存储文件名
            print(file_name)
print(len(small_files))

# 指定文件路径
txt_path = r"E:\dataset\urbanbis\ours\qingdao\2024-09-29-14-21-30\datas.txt"
new_txt_path = r"E:\dataset\urbanbis\ours\qingdao\2024-09-29-14-21-30\filtered_datas.txt"

# 打开并逐行读取原始文件内容
with open(txt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 存储过滤后的行
filtered_lines = []

# 检查每一行是否包含小于600KB的文件名
for line in lines:
    # 如果行中没有任何小于600KB的文件名，就保留该行
    if not any(file_name in line for file_name in small_files):
        filtered_lines.append(line)


# 将过滤后的行写入新的txt文件
with open(new_txt_path, 'w', encoding='utf-8') as new_file:
    new_file.writelines(filtered_lines)

print(f"处理完成，新文件已保存为 {new_txt_path}")
