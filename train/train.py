import os
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import cv2
import torch.nn.init as init
from tqdm import tqdm 
# from models.Unet_Test import UNet2D_Test
from models.Unet_interactive import UNet2D_Test
# from models.UNet_mtl import UNet2D_Test
from matplotlib import pyplot as plt
from loss import DiceLoss
import timeit
from torchvision import transforms
from dataloaders import custom_transforms as tr

from dataloaders.MyDataset import MyDataset
from utils.util import get_next_points

start = timeit.default_timer()

composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.ToTensor()])
T_composed_transforms_tr = transforms.Compose([
        tr.FixedResize(resolutions={'image': (512, 512), 'gt': (512, 512)}, flagvals={'image':cv2.INTER_LINEAR,'gt':cv2.INTER_LINEAR}),
        tr.ToTensor()])

# train_set = MTLDataset('/data/jmc/dataset/glom_data_mtl/train', composed_transforms_tr)
# val_set = MTLDataset('/data/jmc/dataset/glom_data_mtl/val', T_composed_transforms_tr)
train_set = MyDataset('/data/jmc/dataset/wsi/slide_dataset/train', composed_transforms_tr)
val_set = MyDataset('/data/jmc/dataset/wsi/slide_dataset/val', T_composed_transforms_tr)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=True, num_workers=4)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=12, shuffle=False, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Path = 'runs/unet_interactive/'
name = '_epochs.pth'
if not os.path.exists(Path):
        os.makedirs(Path)
model = UNet2D_Test(s_num_classes=1)

model = model.to(device)

# init_weights_he(model)
lr, num_epochs = 5e-5, 80
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

Dice_Loss = DiceLoss().to(device)

for epoch in range(num_epochs):
    model.train()
    
    train_l_seg = 0.0

    for iter, sample in enumerate(tqdm(train_iter, desc=f"Train Epoch {epoch+1}/{num_epochs}", unit="batch")):
        
        image, gt, target = sample['image'], sample['gt'], sample['target']
        
        point = torch.ones(image.shape[0], 2, 3) * -1
        point = point.to(device).to(torch.float32)

        image = image.to(torch.float32)
        image = image.to(device)
        gt = gt.to(torch.float32)
        gt = gt.to(device)
        
        with torch.set_grad_enabled(False):
            with torch.no_grad():
                model.eval()
                preds_s = model(image, point)
                pred = torch.sigmoid(preds_s)
                point = get_next_points(pred, gt, point, 1)
       
        model.train()
        results = model(image, point=point)
        preds_s = results

        l_s = Dice_Loss(preds_s, gt)
       
        optimizer.zero_grad()
        l_s.backward()
        optimizer.step()

        train_l_seg += l_s.item()

    print(f'Train Loss: {train_l_seg/len(train_iter):.4f}')
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for iter, sample in enumerate(tqdm(val_iter, desc=f"Val Epoch {epoch+1}/{num_epochs}", unit="batch")):
            image, gt, target = sample['image'], sample['gt'], sample['target']
            image = image.to(torch.float32)
            image = image.to(device)
            gt = gt.to(torch.float32)
            gt = gt.to(device)
            
            point = torch.ones(image.shape[0], 2, 3) * -1
            point = point.to(device).to(torch.float32)

            preds_s = model(image, point)
            pred = torch.sigmoid(preds_s)
            point = get_next_points(pred, gt, point, 1)
            
            results = model(image, point=point)
            
            preds_s = results

            l_s = Dice_Loss(preds_s, gt)
            val_loss += l_s.item()

    print(f'Val Loss: {val_loss/len(val_iter):.4f}')

    
    if (epoch + 1) % 10 == 0 and (epoch + 1) >= 20:
        torch.save(model.state_dict(), Path+str(epoch+1)+name)


