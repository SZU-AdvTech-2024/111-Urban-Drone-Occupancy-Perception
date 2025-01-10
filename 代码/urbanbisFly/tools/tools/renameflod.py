import os
import re
file = r"/photos/scene/imgs/six\\"
def sort_files(files):
    """
    对文件名列表进行排序，使其按照数字顺序排列。

    :param files: 文件名列表
    :return: 排序后的文件名列表
    """

    # 使用正则表达式提取文件名前的数字部分
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')

    # 根据提取到的数字部分进行排序
    sorted_files = sorted(files, key=extract_number)
    return sorted_files

file_list = os.listdir(file)
file_list = sort_files(file_list)
for i in range(len(file_list)):
    os.rename(file + file_list[i], file + str(i))
