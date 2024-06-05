import os
import random
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import cv2
import torch.nn.init as init
from tqdm import tqdm 

# from models.Unet_Test import UNet2D_Test
from models.Unet_id import UNet2D_Test
# from models.UNet_mtl import UNet2D_Test
from matplotlib import pyplot as plt

from loss import DiceLoss
import timeit
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.MyDataset import MyDataset
from utils.util import *

start = timeit.default_timer()

composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.ToTensor()])
T_composed_transforms_tr = transforms.Compose([
        tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.ToTensor()])


train_set = MyDataset('/data/jmc/dataset/wsi/slide_dataset/train', composed_transforms_tr)
val_set = MyDataset('/data/jmc/dataset/wsi/slide_dataset/val', T_composed_transforms_tr)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=True, num_workers=4)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=12, shuffle=False, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Path = 'runs/unet_interactive_id_best/'
name = '_epochs.pth'
if not os.path.exists(Path):
        os.makedirs(Path)
model = UNet2D_Test(s_num_classes=1)

# pretrained_model_path = '/data/jmc/Omni/MTL/runs/unet_interactive_id_best/120_epochs.pth'
# pretrained_dict = torch.load(pretrained_model_path)
# model.load_state_dict(pretrained_dict)

model = model.to(device)


lr, num_epochs = 5e-4, 120
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
num_iterations = 5
Dice_Loss = DiceLoss().to(device)
BCE_Loss = nn.BCEWithLogitsLoss().to(device)

for epoch in range(num_epochs):
    model.train()
    
    train_l_seg = 0
    it = 0
    for iter, sample in enumerate(tqdm(train_iter, desc=f"Train Epoch {epoch+1}/{num_epochs}", unit="batch")):
        it = random.randint(0, num_iterations)
        image, gt, target = sample['image'], sample['gt'], sample['target']
        
        point = torch.ones(image.shape[0], 20, 3) * -1
        point = point.to(device).to(torch.float32)
       

        image = image.to(torch.float32)
        image = image.to(device)
        gt = gt.to(torch.float32)
        gt = gt.to(device)
        pred = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3]).to(torch.float32).to(device)
        

        for i in range(it):
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    pred_s = model(image, i, point, pred)
                    pred = torch.sigmoid(pred_s)
                    point = get_next_points(pred, gt, point, i+1)


        pred_s = model(image, it, point, pred)
    

        l_s_d = Dice_Loss(pred_s, gt)
        l_s_b = BCE_Loss(pred_s, gt)
    
        l = l_s_d + l_s_b
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_l_seg += l.item() 

    print(f'Train Loss: {train_l_seg / (len(train_iter)):.4f}')
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for iter, sample in enumerate(tqdm(val_iter, desc=f"Val Epoch {epoch+1}/{num_epochs}", unit="batch")):
            it = random.randint(0, num_iterations)
            image, gt, target = sample['image'], sample['gt'], sample['target']
            image = image.to(torch.float32)
            image = image.to(device)
            gt = gt.to(torch.float32)
            gt = gt.to(device)
            
            pred = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3]).to(torch.float32).to(device)

            point = torch.ones(image.shape[0], 20, 3) * -1
            point = point.to(device).to(torch.float32)
        
            # print("train: ", it)
            for i in range(it):
                pred_s = model(image, i, point=point, pred=pred) 
                point = get_next_points(pred, gt, point, i+1)
                # point, re_point = get_next_points_re(pred, gt, point, re_point, i+1)
                
                pred = torch.sigmoid(pred_s)
                

            pred_s = model(image, it, point=point, pred=pred) 
        
            l_s_d = Dice_Loss(pred_s, gt)
            l_s_b = BCE_Loss(pred_s, gt)
            l = l_s_d + l_s_b
        
            val_loss += l.item()

    print(f'Val Loss: {val_loss/(len(val_iter)):.4f}')

    
    
    if (epoch + 1) % 10 == 0 :
        torch.save(model.state_dict(), Path+str(epoch + 1)+name)


