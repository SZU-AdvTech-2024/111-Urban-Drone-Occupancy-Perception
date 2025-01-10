import numpy as np
import torch
def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    # gt_occa = gt_occ.numpy()
    # gt_occ[0][:, :3] -= 1
    # gt_occ[:, 4] -= 1
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float)
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]

    return gt

path = "E:\dataset\\urbanbis\ours\yuehai\\2024-11-14-10-39-41\lidar_point\\lidar_point1731551982000.txt"
point = np.loadtxt(path,skiprows=1,delimiter=',')[:,:4]

ratio = [1,2,4,8]
gt_shape = [[1,64,96,64,96],
            [1,32,48,32,48],
            [1,16,24,16,24],
            [1,8,12,8,12]]
points = torch.from_numpy(point).unsqueeze(0)
gt = multiscale_supervision(points,ratio[3],gt_shape[3])
for i in range(len(ratio)):
    points = torch.from_numpy(point).unsqueeze(0)
    gt = multiscale_supervision(points,ratio[i],gt_shape[i])
    print(1)