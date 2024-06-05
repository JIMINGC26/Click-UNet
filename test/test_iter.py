import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataloaders import custom_transforms as tr
from torch.autograd import Variable

import os
import cv2
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np
from models.Unet_interactive_duel_abl import UNet2D_Test
# from models.Unet_interactive_sole_abl import UNet2D_Test
# from models.Unet_id import UNet2D_Test

from dataloaders.MyDataset import MyDataset
from utils.util import *



def vis_result(imgs, pred, point, mask, path):
    
    predicted_mask = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    img = imgs[0].cpu().numpy() * 255
    img = img.transpose((1, 2, 0))
    point = point.detach().cpu().numpy()
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    
    point = point[0]
    
    mask = np.squeeze(mask[0], axis=0)
    mask[mask < 0] = 0.25
    mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    viz_image = draw_with_blend_and_clicks(img, mask=predicted_mask, 
                                         point=point, radius=6)
    cv2.imwrite(path, viz_image)

def vis_result_cam(imgs, cam, mask, path):
    
    
    img = imgs[0].cpu().numpy() * 255
    img = img.transpose((1, 2, 0))
    mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask[0], axis=0)
    mask[mask < 0] = 0.25
    mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    cam = cam.cpu().numpy()
    cam = np.squeeze(cam, axis=0)
    cam = cam.transpose((1, 2, 0))
    cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.resize(cam, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    viz_image = np.hstack((img, mask, heatmap)).astype(np.uint8)
    # superimposed_img = heatmap * 0.5 + img * 0.5 
    # superimposed_img = superimposed_img.transpose((2, 0, 1))

    
    cv2.imwrite(path, viz_image)

def save_feature_maps_as_heatmaps(feature_maps, layer_name, save_dir):
    feature_maps[layer_name] = torch.mean(feature_maps[layer_name], dim=1)
    features = feature_maps[layer_name].cpu().numpy()
    
    
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # print(features.shape)
    
    fm = features[0]
    fm = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(fm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (512, 512))
        # save_path = os.path.join(save_dir, f"{layer_name}_feature_map_{i}.png")
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_dir, heatmap)
    

feature_maps = {}
def get_feature_map_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output
    return hook


save_dir= "vis/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

transform_test = transforms.Compose([
    tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
    tr.ToTensor()
])
 
test_set = MyDataset('/data/jmc/dataset/glom_data_test/test', transform_test)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet2D_Test(s_num_classes=1)
model.to(DEVICE)
model.load_state_dict(torch.load("runs/abl/unet_interactive_duel_rightTOLeft_abl/110_epochs.pth", map_location=DEVICE))
model.eval()
# model = torch.nn.Sequential(*list(model.children())[:-2])

model.fusionConv.register_forward_hook(get_feature_map_hook("layer3_click"))

total_iou = 0.0
total_accuracy = 0.0
total_dice = 0.0
total_noc = 0.0
num_batches = 0
total_clicks = 0

total_iou_clicks = [0] * 11
total_dice_clicks = [0] * 11

with torch.no_grad():
    for iter, sample in enumerate(tqdm(test_iter, desc="Test")):

        images, gt, target = sample['image'], sample['gt'], sample['target']
        images = images.to(torch.float32)
        images = images.to(DEVICE)
        gt = gt.to(torch.float32)
        gt = gt.to(DEVICE)
        target = target.to(torch.long)
        target = target.to(DEVICE)
        
        point = torch.ones(images.shape[0], 40, 3) * -1.0
        point = point.to(DEVICE).to(torch.float32)
        re_point = torch.ones(images.shape[0], 2, 3) * -1.0
        re_point = re_point.to(torch.float32).to(DEVICE)
        predicted_s = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(torch.float32).to(DEVICE)
        
        # predicted_s, cam = model(images, 0, point=point, pred=predicted_s)
        predicted_s = model(images, 0, point=point, pred=predicted_s)

        predicted_s = torch.sigmoid(predicted_s)
        predicted_masks = (predicted_s > 0.49).float()
        
        
        # save_path = os.path.join(save_dir, f"cam_{iter}_auto.jpg")
        # save_feature_maps_as_heatmaps(feature_maps=feature_maps, layer_name='layer3', save_dir=save_path)
        # vis_result_cam(images, cam, gt, save_path)
        # 计算分割 IoU
        iou = get_iou(predicted_masks, gt)
        # total_iou += iou
        total_iou_clicks[0] += iou

        # 计算Dice
        dice = get_dice(predicted_masks, gt)
        total_dice_clicks[0] += dice
        # if iter in [14, 50, 77, 82]:
        #     save_path = os.path.join(save_dir, f"result_{iter}_auto_{iou:.4f}.jpg")
        #     vis_result(images, predicted_masks, point, gt, save_path)
        # iou = 0.0
        # total_iou_clicks[0] += 0
        # total_dice_clicks[0] += 0
        if iou >= 0.90:
            is_saved = True
        else:
            is_saved = False
        # is_saved = False
        for click_indx in range(1, 21):
            
            if click_indx <= 5:
                id = click_indx
            else:
                id = 5

            point, re_point = get_next_points_re(predicted_s, gt, point, re_point, click_indx)
            # predicted_s, cam = model(images, id, point=point, pred=predicted_s)
            # predicted_s, _ = model(images, id, point=point, pred=predicted_s)
            predicted_s = model(images, id, point=point, pred=predicted_s)
            # predicted_s = model(images, point=point)
            predicted_s = torch.sigmoid(predicted_s)
            predicted_masks = (predicted_s > 0.49).float()
            iou = get_iou(predicted_masks, gt)
            dice = get_dice(predicted_masks, gt)

            if click_indx == 1:
                # save_path = os.path.join(save_dir, f"result_{iter}_click@{click_indx}_{iou:.4f}.jpg")
                save_path = os.path.join(save_dir, f"cam_{iter}_click@{click_indx}.jpg")
                save_feature_maps_as_heatmaps(feature_maps=feature_maps, layer_name='layer3_click', save_dir=save_path)
                save_path = os.path.join(save_dir, f"cam_{iter}_point.jpg")
                
                point_draw = point.detach().cpu().numpy()
                point_draw = point_draw[0]

                point_map = draw_clicks_on_image((512, 512), point_draw, radius=6)
                cv2.imwrite(save_path, point_map)
                break
                # vis_result_cam(images, cam, gt, save_path)
                # vis_result(images, predicted_masks, point, gt, save_path)

            if click_indx <= 10:
                total_iou_clicks[click_indx] += iou
                total_dice_clicks[click_indx] += dice
            if iou >= 0.90 and is_saved is False:
                total_noc += click_indx
                is_saved = True
                if click_indx <= 5 and iter in [14, 50, 77, 82]:
                    save_path = os.path.join(save_dir, f"result_{iter}_iou90_{click_indx}.jpg")
                    vis_result(images, predicted_masks, point, gt, save_path)
                elif iter == 82:
                    save_path = os.path.join(save_dir, f"result_{iter}_iou90_{click_indx}.jpg")
                    vis_result(images, predicted_masks, point, gt, save_path)
                    exit(0)
            elif is_saved is False and click_indx == 20:
                total_noc += click_indx
            

        num_batches += 1

# 计算平均 IoU 和平均准确率
avg_iou = total_iou / num_batches
avg_dice = total_dice / num_batches
avg_accuracy = total_accuracy / num_batches
noc = total_noc / num_batches


print(f"Average Segmentation NoC@90: {noc:.2f}")
print(f"Average Segmentation IoU : {avg_iou:.4f}, and Dice@1: {avg_dice:.4f}") 
print(f"Average Segmentation IoU@0 : {(total_iou_clicks[0]/num_batches):.4f}, and Dice@0: {(total_dice_clicks[0]/num_batches):.4f}") 
print(f"Average Segmentation IoU@1 : {(total_iou_clicks[1]/num_batches):.4f}, and Dice@1: {(total_dice_clicks[1]/num_batches):.4f}") 
print(f"Average Segmentation IoU@2 : {(total_iou_clicks[2]/num_batches):.4f}, and Dice@2: {(total_dice_clicks[2]/num_batches):.4f}")   
print(f"Average Segmentation IoU@3 : {(total_iou_clicks[3]/num_batches):.4f}, and Dice@3: {(total_dice_clicks[3]/num_batches):.4f}") 
print(f"Average Segmentation IoU@4 : {(total_iou_clicks[4]/num_batches):.4f}, and Dice@4: {(total_dice_clicks[4]/num_batches):.4f}") 
print(f"Average Segmentation IoU@5 : {(total_iou_clicks[5]/num_batches):.4f}, and Dice@5: {(total_dice_clicks[5]/num_batches):.4f}") 
print(f"Average Segmentation IoU@6 : {(total_iou_clicks[6]/num_batches):.4f}, and Dice@6: {(total_dice_clicks[6]/num_batches):.4f}") 
print(f"Average Segmentation IoU@7 : {(total_iou_clicks[7]/num_batches):.4f}, and Dice@7: {(total_dice_clicks[7]/num_batches):.4f}") 
print(f"Average Segmentation IoU@8 : {(total_iou_clicks[8]/num_batches):.4f}, and Dice@8: {(total_dice_clicks[8]/num_batches):.4f}") 
print(f"Average Segmentation IoU@9 : {(total_iou_clicks[9]/num_batches):.4f}, and Dice@9: {(total_dice_clicks[9]/num_batches):.4f}") 
print(f"Average Segmentation IoU@10 : {(total_iou_clicks[10]/num_batches):.4f}, and Dice@10: {(total_dice_clicks[10]/num_batches):.4f}") 

