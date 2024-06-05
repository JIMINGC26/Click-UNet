import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools
from utils.DistMap import DistMaps

import sys, os

in_place = False


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(NoBottleneck, self).__init__()
        self.gn1 = nn.GroupNorm(16, inplanes)
        # self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1,1),dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out

class unet2D(nn.Module):
    def __init__(self, layers, s_num_classes=1):
        super(unet2D, self).__init__()
        self.s_num_classes = s_num_classes
        self.inplanes = 128
        self.dist_map = DistMaps(norm_radius=5, spatial_scale=1.0,
                                  cpu_mode=False, use_disks=True)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=1, stride=1, bias=False)
        
        self.layer0_click = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1))
        self.layer1_click = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2))
        self.layer2_click = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2))
        self.layer3_click = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2))

        
        # self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.add_0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(2, 2))
        self.add_1 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(4, 4))

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=(0,0),  dilation=1, bias=False)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=(2, 2))
        self.upsamplex4 = nn.Upsample(scale_factor=(4, 4))

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

        self.x1_resb_add0 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))
        self.x1_resb_add1 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            # nn.Conv2d(32, self.s_num_classes, kernel_size=(1, 1))
            nn.Conv2d(32, 8, kernel_size=(1, 1))
        )

        self.cam_layer_2 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),            
        )

        self.cam_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.softmax = nn.Softmax(dim=-1)
        
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(256, 6)

        self.controller = nn.Conv2d(256+6, 153, kernel_size=1, stride=1, padding=0)   #### change the channel
        

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                # conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                #           weight_std=self.weight_std),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride, padding=(0, 0), dilation=1, bias=False)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)


    def encoding_task(self, click_idx, batch_size):
        N = batch_size
        task_encoding = torch.zeros(size=(N, 6))   #### change the channel
        for i in range(N):
            task_encoding[i, click_idx]=1
        # print(task_encoding, click_idx)
        return task_encoding.cuda("cuda:0")

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts, -1, 1, 1)
                # print("l:",bias_splits[l].shape)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        # print("x:", x.shape)
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(w.shape, b.shape, i)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            # print("x:", x.shape, i)
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    
    def get_point_attention(self, coord_feature, feature):
        b, c, h, w = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]
        coord_feature = coord_feature.view(b, c, -1)
        feature = feature.view(b, c, -1)
        # print(feature.shape, coord_feature.shape)
        result = torch.bmm(feature.transpose(1, 2), feature)
        result = self.softmax(result)
        result = torch.bmm(coord_feature, result)
        result = result.view(b, c, h, w)   
        return result
    
    

    def forward(self, input, click_idx, point=None, pred=None):
        
        coord_features = self.dist_map(input, point).to(pred.device)
        coord_features = torch.cat((coord_features, pred), dim=1)
        coord_features = self.conv2(coord_features)
        
        x = self.conv1(input)
       
        x_c = self.layer0_click(x+coord_features)
        x = self.layer0(x)
        
       
        skip0 = x_c
        
        x_c = self.layer1_click(x_c + x)    
        x = self.layer1(x + skip0)
        
         
        skip1 = x_c

        x_c = self.layer2_click(x_c + x)   
        x = self.layer2(x + skip1)

        
        skip2 = x_c

        x_c = self.layer3_click(x_c + x)
        x = self.layer3(x + skip2)
        
         
        skip3 = x_c      
        
        x_cam = self.cam_layer_2(x)

        x_cam = self.cam_conv(x_cam)


        task_encoding = self.encoding_task(click_idx, x_cam.shape[0])
        # print(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2)#.unsqueeze_(2)
        x_cls = self.GAP(x_cam)
        x_cond = torch.cat([x_cls, task_encoding], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1)#.squeeze_(-1)

        x_cls = torch.flatten(x_cls, 1)

        classifier = self.classifier(x_cls)
        pred_class_idx = torch.argmax(classifier, dim=1)
        weights = self.classifier.weight[pred_class_idx].detach().clone()

        cam = weights.unsqueeze(-1).unsqueeze(-1) * x_cam

        # cam_vis = cam.sum(dim=1, keepdim=True)

        # cam_vis = F.relu(cam_vis)
        # cam_vis = cam_vis - cam_vis.min()
        # cam_vis = cam_vis / cam_vis.max()
        # cam_vis = F.interpolate(cam_vis, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        fusion_feature = self.get_point_attention(x_c, cam)
        
        x = self.layer4(x + x_c + fusion_feature)

        x = self.fusionConv(x)


        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)

        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)

        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)

        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)         # (32, 128, 256)
       
        
        # segment        
        head_inputs = self.precls_conv(x)
        
        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)
        # print(head_inputs.shape)
        logits = self.heads_forward(head_inputs, weights, biases, N)
        # print("logits:", logits.shape)
        logits = logits.reshape(-1, self.s_num_classes, H, W)

        # print(logits.shape)
        # logits = head_inputs.reshape(-1, self.s_num_classes, input.shape[2], input.shape[3])
        
    
        # output = self.seg_head(logits)
        # print(output.shape)
        return logits

def UNet2D_Test(s_num_classes=1):
    print("Using DynConv 8,8,1")
    model = unet2D([1, 2, 2, 2, 2], s_num_classes)
    return model
