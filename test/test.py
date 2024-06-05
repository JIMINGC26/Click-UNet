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
from models.Unet_Test import UNet2D_Test
# from models.UNet_mtl import UNet2D_Test
# from models.Unet_Test import UNet2D_Test
# from models.Unet_interactive_gate import UNet2D_Test
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


    viz_image = draw_with_blend_and_clicks(img, mask=predicted_mask, 
                                         point=point, radius=10)
    cv2.imwrite(path, viz_image)
    



save_dir= "vis/unet_vis"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

transform_test = transforms.Compose([
    tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
    tr.ToTensor()
])
 
test_set = MyDataset('/data/jmc/dataset/glom_data_test/test', transform_test)
# test_set = MTLDataset('/data/jmc/dataset/glom_data_mtl/test', transform_test)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet2D_Test(s_num_classes=1)
model.load_state_dict(torch.load("runs/abl/unet/120_epochs.pth"))
model.eval()
# model = torch.nn.Sequential(*list(model.children())[:-2])
model.to(DEVICE)

total_iou = 0.0
total_accuracy = 0.0
total_dice = 0.0
total_noc = 0.0
num_batches = 0
total_clicks = 0

with torch.no_grad():
    for iter, sample in enumerate(tqdm(test_iter, desc="Test")):

        images, gt, target = sample['image'], sample['gt'], sample['target']
        images = images.to(torch.float32)
        images = images.to(DEVICE)
        gt = gt.to(torch.float32)
        gt = gt.to(DEVICE)
        
        # print(images.shape, gt.shape)
        # exit()

        point = torch.ones(images.shape[0], 40, 3) * -1.0
        point = point.to(DEVICE).to(torch.float32)

        # 预测
        # segmentation_outputs, classification_outputs = model(images)['segment'], model(images)['classify']
        # cam = model(images)['cam']
        # segmentation_outputs = model(images)
        segmentation_outputs = model(images)
        segmentation_outputs = torch.sigmoid(segmentation_outputs)
        predicted_masks = (segmentation_outputs > 0.49).float()  # 假设0.5是阈值
        
        save_path = os.path.join(save_dir, f"result_{iter}_auto.jpg")
        vis_result(images, predicted_masks, point, gt, save_path)
        # print(predicted_masks.sum())
        # save_path = os.path.join(save_dir, f"seg_result_{iter}.jpg")
        # vis_result(images, predicted_masks, cam, save_path)

        # segmentation_result =  predicted_masks.cpu().numpy().reshape(images.shape[2], images.shape[3])  # 假设是2D分割图
        # # 保存分割结果
        # save_path = os.path.join(save_dir, f"seg_result_{iter}.jpg")
        # result_image = Image.fromarray((segmentation_result * 255).astype('uint8'))  # 根据需要调整
        # result_image.save(save_path)


        # 计算分割 IoU
        iou = get_iou(predicted_masks, gt)
        total_iou += iou

        # 计算Dice
        dice = get_dice(predicted_masks, gt)
        total_dice += dice

 
        num_batches += 1

# 计算平均 IoU 和平均准确率
avg_iou = total_iou / num_batches
avg_dice = total_dice / num_batches
avg_accuracy = total_accuracy / num_batches
noc = total_noc / num_batches

print(f"Average Segmentation IoU: {avg_iou:.2f}")
print(f"Average Segmentation NoC@90: {noc:.2f}")
print(f"Average Segmentation Dice: {avg_dice:.2f}")
print(f"Average Classification Accuracy: {avg_accuracy:.2f}") 
  
