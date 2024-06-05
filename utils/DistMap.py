import torch
from torch import nn as nn
import numpy as np

# 点击的距离映射
class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks

    def get_coord_features(self, points, batchsize, rows, cols):
        
        num_points = points.shape[1] // 2
        points = points.contiguous().view(-1, points.size(2))
        # 沿第二维切分为长度为2和1的两个新的数组
        points, points_order = torch.split(points, [2, 1], dim=1)

        invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
        row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
        # 将张量按指定维度堆叠，按行方向扩充，第一维重复其长度次（即生成候选点数量个空白张量
        coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

        add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
        coords.add_(-add_xy)
        if not self.use_disks:
            coords.div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)

        coords[:, 0] += coords[:, 1]
        coords = coords[:, :1]

        coords[invalid_points, :, :, :] = 1e6

        coords = coords.view(-1, num_points, 1, rows, cols)
        coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
        coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])
