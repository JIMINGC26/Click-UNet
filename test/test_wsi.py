import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataloaders import custom_transforms as tr

import time
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from models.Unet_id import UNet2D_Test
# from models.Unet_interactive_nocam import UNet2D_Test
# from models.Unet_interactive_duel_abl import UNet2D_Test
from utils.util import *


def sliding_window(model, device, image, mask, window_size, stride, transform_test=None, points=None, is_refine=False, pred=None):
    h, w = image.shape[:2]
    noc = 0
    
    if (is_refine):
        new_w, new_h = w, h
    else:
        new_w = (w // crop_size[1] + 1) * crop_size[1]
        new_h = (h // crop_size[0] + 1) * crop_size[0]
    new_image = np.zeros((new_h, new_w, 3))
    new_mask = np.zeros((new_h, new_w))
    new_image[:h, :w, :] = image
    new_mask[:h, :w] = mask
    print(new_h, new_w)
    output = np.zeros((h, w))
    
    for y in range(0, new_h  - window_size[0] + 1, stride):
        for x in range(0, new_w  - window_size[1] + 1, stride):
            
            window = new_image[y:y+window_size[0], x:x+window_size[1],:].copy()
            mask_window = new_mask[y:y+window_size[0], x:x+window_size[1]].copy()
            
            sample = {'image': window.copy(), 'gt': mask_window}
            if transform_test is not None:
                 sample = transform_test(sample)
            img, gt = sample['image'], sample['gt']
            mask = torch.tensor(mask_window)
            
            img = img.unsqueeze(dim=0)
            gt = gt.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)
            # print("before input:",img.shape, gt.shape)
            
            img = img.to(device).to(torch.float32)
            gt = gt.to(device).to(torch.float32)
            pred = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3]).to(torch.float32).to(device)
                
            point = torch.ones(img.shape[0], 40, 3) * -1.0
            point = point.to(device).to(torch.float32)
            
            if is_refine:
                # segmentation_outputs = model(img, 0, point, pred)
                # pred = torch.sigmoid(segmentation_outputs)
                if points[0, 0, 0] != -1:
                    for i in range(2):
                        point[0, 0, i] = points[0, 0, i]  # * 512 / 3000
                    point[0, 0, 2] = 1
                else:
                    for i in range(2):
                        point[0, point.shape[1]-1, i] = points[0, 1, i] # * 512 / 3000
                    point[0, point.shape[1]-1, 2] = 1
            
            if not is_refine:
                pred_out = predict(model, image=img, mask=gt, point=point, pred=pred)
            else:
                pred_out,  noc = predict_re(model, image=img, mask=gt, point=point, pred=pred)

            pred_out = pred_out.cpu().numpy()
            pred_out = np.squeeze(pred_out, axis=0)
            pred_out = np.transpose(pred_out, (1, 2, 0))
            pred_out = cv2.resize(pred_out, window_size)
            

            if ( y + window_size[0] > h and x + window_size[1] > w):
                output[y:h, x:w] = np.maximum(output[y:h, x:w], pred_out[:h-y, :w-x])
            elif ( y + window_size[0] > h):
                output[y:h, x:x + window_size[1]] = np.maximum(output[y:h, x:x + window_size[1]], pred_out[:h-y, :window_size[1]])
            elif (x + window_size[1] > w ):
                output[y:y + window_size[0], x:w] = np.maximum(output[y:y + window_size[0], x:w], pred_out[:window_size[0], :w-x])
            else:
                output[y:y+window_size[0], x:x+window_size[1]] = np.maximum(output[y:y+window_size[0], x:x+window_size[1]], 
                                                                        pred_out[:window_size[0], :window_size[1]])   
    if not is_refine:
        output = remove_small_regions(output)

    return output, noc

def remove_small_regions(mask, min_area_threshold=3000):
    # 寻找连通区域
    mask = np.uint8(mask)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    # 过滤掉小区域
    filtered_mask = np.zeros_like(mask)
    for i, stat in enumerate(stats):
        if i == 0:  # 忽略背景区域
            continue
        area = stat[4]  # 连通区域的面积
        if area >= min_area_threshold:
            # 将大区域添加到结果掩码中
            filtered_mask[labels == i] = 1

    return filtered_mask

def predict(model, image, mask, point, pred):
    with torch.no_grad():
        segmentation_outputs = model(image, 0, point, pred)
        # print(image.shape, point.shape)
        # segmentation_outputs = model(image, point, pred)
        segmentation_outputs = torch.sigmoid(segmentation_outputs)
        predicted_masks = (segmentation_outputs > 0.49).float()
        # print(f"Predict iou:{get_iou(predicted_masks, mask):.2f}")
    return predicted_masks

def predict_re(model, image, mask, point, pred):
    with torch.no_grad():
        segmentation_outputs = model(image, 1 , point, pred)
        pred = torch.sigmoid(segmentation_outputs)
        predicted_masks = (pred > 0.49).float()
       
        
        iou = get_iou(predicted_masks, mask)
        
        click_idx = 1
    
        while iou <= 0.900:
            if click_idx >= 20:
                break
            click_idx += 1
            if click_idx >= 5:
                id = 5
            else:
                id = click_idx
            point = get_next_points(pred, mask, point, click_idx)
            segmentation_outputs = model(image, id, point, pred)
            # segmentation_outputs = model(image, point, pred)
            pred = torch.sigmoid(segmentation_outputs)
            predicted_masks = (pred > 0.49).float()
            iou = get_iou(predicted_masks, mask)
        print("Local iou:", iou)

    return predicted_masks, click_idx

def save_results(preds, fname, path):
    predicted_mask = preds.clone().numpy()
    
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    predicted_mask = np.uint8(predicted_mask) * 255
    predicted_mask = predicted_mask.transpose((1, 2, 0))
    
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.IMREAD_GRAYSCALE)
    predicted_mask = cv2.resize(predicted_mask, (int(predicted_mask.shape[1] / 512 * 3000), int(predicted_mask.shape[0] / 512 * 3000)), interpolation=cv2.INTER_LINEAR)
    save_path = os.path.join(path, f"result_{fname}")
    cv2.imwrite(save_path, predicted_mask)

def save_result_with_click(imgs, pred, point, path):
    predicted_mask = pred.detach().cpu().numpy()
    
    img = imgs[0] * 255
    img = img.transpose((1, 2, 0))
    point = point.detach().cpu().numpy()
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    # predicted_mask = np.uint8(predicted_mask) * 255
    # predicted_mask = predicted_mask.transpose((1, 2, 0))
    # print(predicted_mask.shape)
    # predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    # predicted_mask = np.expand_dims(predicted_mask, axis=-1)
    
    point = point[0]


    viz_image = draw_with_blend_and_clicks(img, mask=predicted_mask, 
                                         point=point, radius=10)
    cv2.imwrite(path, viz_image)


def refine(image, mask, point, crop_size):
    coord = get_coord(point)
    # print(image.shape)
    crop_location = get_crop_locations(coord, crop_size, image.shape[2], image.shape[3])
    # print(crop_location, coord, crop_size, image.shape[2], image.shape[3])
    cropped_image = crop(image, crop_location, crop_size[0], crop_size[1])
    cropped_mask = crop(mask, crop_location, crop_size[0], crop_size[1])
    # cropped_preds = crop(predicted_mask, crop_location, crop_size[0], crop_size[1])
    new_point = justify_points(point, crop_location)
    # print(new_point, point, crop_location)
    
    return cropped_image, cropped_mask, new_point, crop_location

if __name__ == "__main__":
    # 读取大图和掩码
    root = '/data/jmc/dataset/wsi/dataset/test'
    img_dir = root+'/images'
    mask_dir = root+'/masks'
    save_dir = "vis/unet_wsi_cam_abl"
    # log 
    txt_path = ""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置滑动窗口参数
    window_size = (512, 512)
    stride = 256
    factor = 512 / 3000
    
    transform_test = transforms.Compose([
        tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.ToTensor()])
    trans = transform_test
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet2D_Test(s_num_classes=1)
    model.to(DEVICE)
    model.load_state_dict(torch.load("runs/unet_interactive_id_best/120_epochs.pth", map_location=DEVICE))
    model.eval()
    

    crop_size = (512, 512)
    total_iou = 0
    total_dice = 0
    total_iou_re = 0
    total_dice_re = 0
    total_click = 0
    num = 0
    
    for root, _, fnames in os.walk(img_dir):
        for fname in sorted(fnames):
            with open(txt_path, "a") as file:
                if fname.endswith(".png"): #查看文件是否是支持的可扩展类型，是则继续
                    
                    print(f"processing, {fname}!")
                    start_time = time.perf_counter()
                    file.write(f"Picture Name: {fname}\n")
                    click_idx = 0
                    image_path = os.path.join(root, fname)
                    mask_path = os.path.join(mask_dir, fname.split('.')[0]+"_mask."+fname.split('.')[1])
                    
                    original_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                    
                    original_image = cv2.resize(original_image, (int(original_image.shape[1] * 512 / 3000),  int(original_image.shape[0] * 512 / 3000)), interpolation=cv2.INTER_LINEAR)
                    
                    original_image = original_image / 255.0
                    original_image = np.array(original_image, dtype=np.float32)
                    
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                    mask = cv2.resize(mask, (int(mask.shape[1] * 512 / 3000),  int(mask.shape[0] * 512 / 3000)), interpolation=cv2.INTER_LINEAR)
                    
                    mask = mask / 255.0
                    mask = np.array(mask, dtype=np.float32)
                    
                    
                    preds, _ = sliding_window(model, DEVICE, original_image, mask, window_size, 
                                            stride, transform_test=trans)
                   
                    
                    mask = np.expand_dims(mask, axis=0)
                    mask = np.expand_dims(mask, axis=0)
                    
                    
                    
                    kernel = np.ones((10, 10), np.uint8)
                    preds_np = cv2.morphologyEx(preds, cv2.MORPH_OPEN, kernel)
                    preds_np = np.expand_dims(preds_np, axis=0)
                    preds_np = np.expand_dims(preds_np, axis=0)

                    mask = torch.tensor(mask)
                    preds = torch.tensor(preds_np)
                    
                    # save_results(preds, fname.split('.')[0]+"_auto."+fname.split('.')[1], save_dir)
                    
                    iou = get_iou(preds, mask)
                    dice = get_dice(preds, mask)

                    total_iou += iou
                    total_dice += dice

                    print(f"Before  iou:{iou:.3f}, dice:{dice:.3f}")
                    file.write(f"Before refine iou:{iou:.3f}, dice:{dice:.3f}\n")
                    points = torch.ones(1, 40, 3) * -1.0
                    points = points.to(DEVICE).to(torch.float32)

                    point = torch.ones(1, 2, 3) * -1.0
                    point = point.to(DEVICE).to(torch.float32)

                    original_image = np.expand_dims(original_image, axis=0)
                    original_image = np.transpose(original_image, (0, 3, 1, 2))
                    total_noc_one = 0
                    while round(iou, 4) < 0.90:
                        if click_idx >= 20:
                            break

                        click_idx += 1
                        
                        points, point = get_next_points_re(preds, mask, points, point, click_idx)
                        
                        cropped_img, cropped_mask, cropped_point, crop_location = refine(original_image, mask, 
                                                                                            point, crop_size)
                        
                        cropped_img = cropped_img.numpy()
                        cropped_mask = cropped_mask.numpy()
                        cropped_img = np.squeeze(cropped_img, axis=0)
                        cropped_mask = np.squeeze(cropped_mask, axis=0)
                        cropped_mask = np.squeeze(cropped_mask, axis=0)
                        
                        cropped_img = np.transpose(cropped_img, (1, 2, 0))
                        cropped_preds, noc = sliding_window(model, DEVICE, cropped_img, cropped_mask, window_size, 
                                            stride, transform_test=trans, points=cropped_point, is_refine=True)
                        
                        total_click += noc

                        cropped_preds = np.expand_dims(cropped_preds, axis=0)
                        cropped_preds = np.expand_dims(cropped_preds, axis=0)
                        cropped_preds = torch.tensor(cropped_preds)
                        
                        paste(cropped_preds, preds, crop_location)
                    
                        dice = get_dice(preds, mask)
                        print("refine iou dice:",iou, dice, noc)
                        file.write(f"refine, iou:{iou:.3f}, dice:{dice:.3f}, noc:{noc}\n")
                        #save_results(preds, f'{click_idx}_{fname}', save_dir)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print(f"函数运行时间: {elapsed_time:.2f} 秒")
                    
                    save_path = os.path.join(save_dir, f"result_{fname.split('.')[0]}_{iou:.3f}_final.jpg")
                    

                    print(f"After iou:{iou:.3f}, dice:{dice:.3f}")
                    file.write(f"After refine, noc:{iou:.3f}, dice:{dice:.3f}\n")
                    file.write(f"函数运行时间: {elapsed_time:.2f} 秒\n")
                    
                    
                    num += 1
    
    with open(txt_path, "a") as file:
        file.write(f"Average Ori IoU:{(total_iou / num) :.3f}\n")
        file.write(f"Average Ori Dice:{(total_dice / num) :.3f}\n")
        file.write(f"Average Refine IoU:{(total_iou_re / num) :.3f}\n")
        file.write(f"Average Refine Dice:{(total_dice_re / num) :.3f}\n")
        file.write(f"Average NoC90%:{(total_click / num) :.2f}\n")

    print(f"Average IoU:{(total_iou / num) :.3f}")
    print(f"Average dice:{(total_dice / num) :.3f}")
    print(f"Average NoC90%:{(total_click / num):.2f}")