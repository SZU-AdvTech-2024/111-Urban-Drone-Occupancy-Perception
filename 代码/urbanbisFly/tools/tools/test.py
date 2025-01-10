import numpy as np
import torch
from matplotlib import pyplot as plt



def point_sampling( reference_points, pc_range,lidar2img):
    # 5 4 4
    lidar2img = np.expand_dims(lidar2img, axis=0)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

    reference_points = reference_points.clone()
    # 由0-1映射到真实坐标系当中
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]

    # 齐次
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # 雷达坐标系转图片坐标系
    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    volume_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= 1504
    reference_points_cam[..., 1] /= 800
    # 移除超出相机范围的像素
    volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
                   & (reference_points_cam[..., 1:2] < 1.0)
                   & (reference_points_cam[..., 0:1] < 1.0)
                   & (reference_points_cam[..., 0:1] > 0.0))


    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # num_cam, B, num_query, D, 3
    volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    true_count = volume_mask.sum().item()  # 使用 .item() 转换为 Python 数字
    # print("True 的数量:", true_count)
    return reference_points_cam, volume_mask


def point_sampling_ours( reference_points, pc_range,lidar2img):
    # 5 4 4
    lidar2img = np.expand_dims(lidar2img, axis=0)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

    reference_points = reference_points.clone()
    # 由0-1映射到真实坐标系当中
    # reference_points[..., 0:1] = 1.5 * reference_points[..., 0:1] - pc_range[0]
    # reference_points[..., 1:2] = 1.5 * reference_points[..., 1:2] + pc_range[4]
    # reference_points[..., 2:3] = 1.5 * reference_points[..., 2:3] - pc_range[2]

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]
    # np.savetxt("aa.txt",reference_points[0][0].cpu().numpy(),delimiter=",")
    # 齐次
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    # reference_points = reference_points.permute(1, 0, 2, 3)
    # D, B, num_query = reference_points.size()[:3]
    # num_cam = lidar2img.size(1)
    #
    # reference_points = reference_points.view(
    #     D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # # 雷达坐标系转图片坐标系
    # lidar2img = lidar2img.view(
    #     1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
    #
    # reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
    #                                     reference_points.to(torch.float32)).squeeze(-1)
    # reference_points_cam = reference_points_cam[...,:3]
    #
    # # 提取 x, y, z
    # x = reference_points_cam[..., 0]  # 取最后一维的第一个值 (x)
    # y = reference_points_cam[..., 1]  # 取最后一维的第二个值 (y)
    # z = reference_points_cam[..., 2]  # 取最后一维的第三个值 (z)
    #
    # # 避免 z 为 0 时的除法问题
    # eps = 1e-5
    # z = torch.where(z == 0, torch.ones_like(z) * eps, z)
    #
    # # 分别将 x 和 y 除以 z
    # x_div_z = x / z
    # y_div_z = y / z
    # u_valid = (x_div_z >= 0) & (x_div_z < 1504)
    # v_valid = (y_div_z >= 0) & (y_div_z < 800)
    # valid_mask = u_valid & v_valid
    #
    # # 将 x_div_z, y_div_z 和 z 重新拼接回原来的张量
    # reference_points_cam = torch.stack((x_div_z, y_div_z, z), dim=-1)[valid_mask]

    print(11)
    # eps = 1e-5
    #
    # volume_mask = (reference_points_cam[..., 2:3] > eps)
    # reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #     reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    #
    # reference_points_cam[..., 0] /= 1504
    # reference_points_cam[..., 1] /= 800
    # # 移除超出相机范围的像素
    # volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
    #                & (reference_points_cam[..., 1:2] < 1.0)
    #                & (reference_points_cam[..., 0:1] < 1.0)
    #                & (reference_points_cam[..., 0:1] > 0.0))


    # reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # num_cam, B, num_query, D, 3
    # volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    # true_count = volume_mask.sum().item()  # 使用 .item() 转换为 Python 数字
    # # print("True 的数量:", true_count)
    # return reference_points_cam, volume_mask
def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float):
    # 32 60 60
    zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                        device=device).view(Z, 1, 1).expand(Z, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                        device=device).view(1, 1, W).expand(Z, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                        device=device).view(1, H, 1).expand(Z, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
    ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
    return ref_3d

def vis(points):
    # 可视化三维参考点
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 提取 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 绘制 3D 散点图
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('3D Reference Points Visualization')
    plt.show()

if __name__ == '__main__':
    # 调用函数生成参考点
    H = 32  # 高度
    W = 60  # 宽度
    Z = 60  # 深度
    bs = 1  # batch size
    pc_range = [-90, 0, -90, 90, 96, 90]

    lidar2img = np.array(
        [
            [
                [750.00000000, 649.51905284, 375.00000000, 0.00000000],
                [0.00000000, 750.91016151, -500.61455166, 0.00000000],
                [0.00000000, 0.86602540, 0.50000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000]
            ],
            [
                [-375.00000000, 649.51905284, 750.00000000, 0.00000000],
                [500.61455166, 750.91016151, -0.00000000, 0.00000000],
                [-0.50000000, 0.86602540, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000]
            ],
            [
                [375.00000000, 649.51905284, -750.00000000, 0.00000000],
                [-500.61455166, 750.91016151, -0.00000000, 0.00000000],
                [0.50000000, 0.86602540, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000]
            ],
            [
                [750.00000000, 750.00000000, 0.00000000, 0.00000000],
                [0.00000000, 400.00000000, -809.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000]
            ],
            [
                [-750.00000000, 649.51905284, -375.00000000, 0.00000000],
                [0.00000000, 750.91016151, 500.61455166, 0.00000000],
                [-0.00000000, 0.86602540, -0.50000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000]
            ]
        ]
    )
    # 启动函数并生成参考点
    reference_points = get_reference_points(H, W, Z, bs=bs)
    # 从生成的参考点中提取第一个batch的坐标数据
    # points = reference_points[0, 0].cpu().numpy()  # 形状: (Z*H*W, 3)
    # np.savetxt("test.txt",points,delimiter=",")
    reference_points_cam, volume_mask = point_sampling(reference_points, pc_range, lidar2img)
    # reference_points_cam = reference_points_cam[volume_mask==True]
    mask = volume_mask[4]
    mask_valid = mask.squeeze(-1).unsqueeze(1)
    true_count = mask_valid.sum().item()
    # 使用布尔索引保留 mask 为 True 的点
    filtered_points = reference_points[mask_valid]  # 筛选后的点云，形状: [N, 3]，N 是符合条件的点数量
    np.savetxt("test.txt",filtered_points.cpu().numpy(),delimiter=",")
    print(11)
    # points = reference_points_cam[0, 0].cpu().numpy()  # 形状: (Z*H*W, 3)
    # np.savetxt("test.txt",points,delimiter=",")