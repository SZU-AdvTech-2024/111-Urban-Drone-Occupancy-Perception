import json
import matplotlib.pyplot as plt

def get_json_data(path):
    # 假设数据在一个名为data.json的文件中，你可以根据实际文件名修改
    with open(path, 'r') as file:
        lines = file.readlines()[1:]  # 跳过第一行，获取剩余行数据
    visa = True
    iou_values = []
    loss_occ_0 = []
    loss_occ_1 = []
    loss_occ_2 = []
    loss_occ_3 = []
    loss = []
    for line in lines:
        data = json.loads(line.strip())
        data
        if data["mode"] == 'val':
            iou_value = data["IoU"]
            if iou_value is not None:
                iou_values.append(iou_value)
        else:
            loss_occ_0.append(data["loss_occ_0"])
            loss_occ_1.append(data["loss_occ_1"])
            loss_occ_2.append(data["loss_occ_2"])
            loss_occ_3.append(data["loss_occ_3"])
            loss.append(data["loss"])
    return iou_values,loss_occ_0,loss_occ_1,loss_occ_2,loss_occ_3,loss

def show(list1,list2,zhuti):
    # 设置横坐标，这里简单使用索引作为横坐标，你也可根据实际情况替换，比如迭代次数等
    x = range(len(list1))

    # 绘制第一条折线
    plt.plot(x, list1, label='Surroundocc', marker='o')

    # 绘制第二条折线
    plt.plot(x, list2, label='Droneocc', marker='s')

    # 添加标题和坐标轴标签
    plt.title(zhuti)
    plt.xlabel('index')
    plt.ylabel('Value')

    # 添加图例，方便区分两条折线代表的数据
    plt.legend()

    # 展示图形
    plt.show()
path_v2 =  "E:\project\SurroundOcc-main\work_dirs\surroundocc_urbanbis_v2\\20241203_142541.log.json"
path_v1 = "E:\project\SurroundOcc-main\work_dirs\surroundocc_urbanbis_lihuyuehai\\20241203_142158.log.json"

iou_values_v1,loss_occ_0_v1,loss_occ_1_v1,loss_occ_2_v1,loss_occ_3_v1,loss_v1  = get_json_data(path_v1)
iou_values_v2,loss_occ_0_v2,loss_occ_1_v2,loss_occ_2_v2,loss_occ_3_v2,loss_v2  = get_json_data(path_v2)

show(iou_values_v1,iou_values_v2,"val_iou")
show(loss_occ_0_v1,loss_occ_0_v2,"loss_occ_0")
show(loss_occ_1_v1,loss_occ_1_v2,"loss_occ_1")
show(loss_occ_2_v1,loss_occ_2_v2,"loss_occ_2")
show(loss_occ_3_v1,loss_occ_3_v2,"loss_occ_3")
show(loss_v1,loss_v2,"loss")



