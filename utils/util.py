import numpy as np
import cv2
import torch
from pathlib import Path
from torch.nn import functional as F
import os, sys
sys.path.append(os.getcwd())
from utils.serialization import load_model

def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        # 计算非零像素到最近值为零的像素的距离
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        # print(f"positive:{is_positive}")
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
        # print(f"points:{bindx, points[bindx, :, :]}")
    return points


def get_next_points_re(pred, gt, points, refine_point, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().detach().numpy()[:, 0, :, :]
    # 返回的gt为bool值组成的数组
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    # 实际为真，预测为假，找到将真判断为假的区域
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    # 实际为假，预测为真，找到将假判断为真的区域
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    # 填充
    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()
    refine_point = remove_all(refine_point)

    for bindx in range(fn_mask.shape[0]):
        # 计算非零像素到最近值为零的像素的距离
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt 
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        # 返回符合的元素坐标
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            # coords = indices[np.random.randint(0, len(indices))]
            coords = indices[0]
            if is_positive:
                # 正向点击
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
                refine_point[bindx, 0, 0] = float(coords[0])
                refine_point[bindx, 0, 1] = float(coords[1])
                refine_point[bindx, 0, 2] = float(click_indx)
            else:
                # 负向点击
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
                refine_point[bindx, 1, 0] = float(coords[0])
                refine_point[bindx, 1, 1] = float(coords[1])
                refine_point[bindx, 1, 2] = float(click_indx)
            # print(f"indices:{len(indices)}")
                   
        # print(f"refine_point:{bindx, points[bindx, :, :]}")
    return points, refine_point

def remove_all(points):    
    for i in range(points.shape[0]):
        points[i] = points[i] * 0.0 - 1.0
    return points

def get_iou(predicted_masks, gt):
    intersection = (predicted_masks * gt).sum(dim=(1, 2, 3))
    union = predicted_masks.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()

def get_dice(predicted_masks, gt):
    intersection_d = (predicted_masks * gt).sum(dim=(1, 2, 3))
    union_d = predicted_masks.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = 2.0 * intersection_d / union_d
    return dice.mean().item()

def draw_clicks_on_image(image_size, click_coords, radius, color=(255, 255, 255)):
    """
    根据点击坐标在黑色背景图上绘制点击点。
    
    :param image_size: 图像的尺寸 (宽度, 高度)
    :param click_coords: 点击坐标列表 [(x1, y1), (x2, y2), ...]
    :param radius: 点击点的半径
    :param color: 点击点的颜色 (默认白色)
    :return: 带有点击点的图像
    """
    # 创建黑色背景图像
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    point_pos = click_coords[:20]
    point_neg = click_coords[20:]
    # 在图像上绘制点击点
    image = draw_points(image, point_pos, color=(0, 255, 0), radius=radius)
    image = draw_points(image, point_neg, color=(0, 127, 255), radius=radius)
    
    return image


def draw_points(image, points, color, radius=6):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 6, 1: 6, 2: 6}[p[2]] if p[2] < 3 else 6
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image



def get_palette(num_cls):

    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0
        
        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            # print(palette[j*3 + 0], palette[j*3 + 1], palette[j*3 + 2])
            i = i + 1
            lab >>= 3
    # exit(0)
    return palette.reshape((-1, 3))


def draw_with_blend_and_clicks(img, mask=None, alpha=0.8, point=None, pos_color=(0, 255, 0),
                               neg_color=(0, 127, 255), radius=6):
    result = img.copy()
    # print(result.shape, mask.shape)
    if mask is not None:
        palette = get_palette(int(np.max(mask)) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]
        # rgb_mask = np.squeeze(rgb_mask, axis=2)

        mask_region = (mask > 0).astype(np.uint8)
        # print(result.shape, mask_region.shape, rgb_mask.shape)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)
        
        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)
        # print(result.shape)
    if point is not None:

        result = draw_points(result, point[:20], pos_color, radius=radius)
        result = draw_points(result, point[20:], neg_color, radius=radius)

    return result

def get_coord(points):
    coords = []    
    device = points.device
    
    for b in range(points.size(0)):
        for i in range(2):
            
            if points[b, i, 0] != -1 and points[b, i, 1] != -1:
                # 会出现只有三个点击的情况
                coord = [points[b, i, 0], points[b, i, 1], float(i)]
                coords.append(coord)
        if points[b, 0, 0] == -1 and points[b, 0, 1] == -1 and points[b, 1, 0] == -1 and points[b, 1, 1] == -1:
            coord = [-1., -1. , 0]
            coords.append(coord)

    coords = torch.tensor(coords).to(device)
    # print(f"coords:{coords.shape}")
    return coords


def crop(x, crop_locations, crop_h, crop_w):
    # print(type(x))
    if type(x) == np.ndarray:
        b, _, h, w = x.shape
    else:
        b, _, h, w = x.size()

    y = []
    for i in range(b):
        #  print(f"location:{crop_locations.shape}")
        if crop_locations[i, 0] != -h:
            y0, x0, y1, x1 = crop_locations[i, 0].item(), crop_locations[i, 1].item(), crop_locations[i, 2].item(), crop_locations[i, 3].item()
        else:
            y0, x0, y1, x1 = 0, 0, int(crop_h), int(crop_w)
        # print(f"locations:{y0, y1, x0, x1}")
        cropped_x = x[i, :, y0: y1, x0: x1]
        # print(f"x:{cropped_x.shape}")
        # exit()
        # print(f"size:{}")
        # print(cropped_x.shape)
        cropped_x = torch.tensor(cropped_x)
        
        cropped_x = F.interpolate(cropped_x.unsqueeze(0), size=(int(crop_h), int(crop_w)), mode='bilinear', align_corners=True)
        y.append(cropped_x)

    # print(f"y.len:{len(y)}")
    # exit()
    
    y = torch.stack(y)
    y = torch.squeeze(y, 1)
    return y

def paste(patches, x, crop_locations): 
    if type(x) == np.ndarray:
        batch_size, _, h_1, w_1 = x.shape
        
    else:
        batch_size, _, h_1, w_1 = x.size()
    
    for i in range(batch_size):
        if crop_locations[i, 0] != -(h_1):
            y0, x0, y1, x1 = crop_locations[i, 0].item(), crop_locations[i, 1].item(), crop_locations[i, 2].item(), crop_locations[i, 3].item()
            h, w = y1 - y0, x1 - x0
            # print(f"re_output:{patches.shape}, h:{h}, w:{w}")
            patches = F.interpolate(patches, size=(int(h), int(w)), mode='bilinear', align_corners=True)
            x[i, :, y0 : y1, x0 : x1] = patches[i]
        else:
            break
    return x

def get_crop_locations(re_point, crop_sizes, h, w):
    crop_h, crop_w = crop_sizes
    crop_h, crop_w = int(crop_h / 2 + 0.5), int(crop_w / 2 + 0.5)
    crop_locations = torch.zeros(re_point.size(0), 4).to(re_point.device)

    # point存储方式不同
    for i in range(re_point.shape[0]):
        if re_point[i, 0] != -1 and re_point[i, 1] != -1:
            crop_locations[i, 0] = re_point[i, 0] - crop_h
            crop_locations[i, 1] = re_point[i, 1] - crop_w
            crop_locations[i, 2] = re_point[i, 0] + crop_h
            crop_locations[i, 3] = re_point[i, 1] + crop_w
        else:
            crop_locations[i, 0] = -h
            crop_locations[i, 1] = -w
            crop_locations[i, 2] = -h
            crop_locations[i, 3] = -w
    
    crop_locations = limit_crop_locations(crop_locations, h, w)

    crop_locations = crop_locations.long()
    return crop_locations


def limit_crop_locations(crop_locations, h, w):
    # zero_tensor = torch.zeros_like(crop_locations[:, 0]).to(crop_locations.device)
    # one_tensor = torch.ones_like(crop_locations[:, 0]).to(crop_locations.device)

    for i in range(crop_locations.shape[0]):
        # print(f"x:{crop_locations[i,0], crop_locations[i,1]}")
        if crop_locations[i, 0] > -h and crop_locations[i, 1] > -w:
            # crop_locations[i, 0] = torch.maximum(crop_locations[i, 0], zero_tensor)
            # crop_locations[i, 1] = torch.maximum(crop_locations[i, 1], zero_tensor)
            # crop_locations[i, 2] = torch.minimum(crop_locations[i, 2], h * one_tensor)
            # crop_locations[i, 3] = torch.minimum(crop_locations[i, 3], w * one_tensor)
            crop_locations[i, 0] = crop_locations[i, 0] if crop_locations[i, 0] > 0 else 0
            crop_locations[i, 1] = crop_locations[i, 1] if crop_locations[i, 1] > 0 else 0
            crop_locations[i, 2] = crop_locations[i, 2] if crop_locations[i, 2] < h else h
            crop_locations[i, 3] = crop_locations[i, 3] if crop_locations[i, 3] < w else w
        # print(f"x:{crop_locations[i,0], crop_locations[i,1]}")
    return crop_locations

def justify_points(points, crop_locations):
    results = points.clone().detach()
    for b in range(points.size(0)):
            for i in range(2):
                if points[b, i, 0] != -1:
                    results[b, i ,0] = points[b, i, 0] - crop_locations[b, 0]
                    results[b, i ,1] = points[b, i, 1] - crop_locations[b, 1]
    results = results.long()
    return results


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    # print(state_dict)
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model
