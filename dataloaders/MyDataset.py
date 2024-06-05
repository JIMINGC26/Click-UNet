import os
import numpy as np
import sys
import cv2
import re
import torch
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from skimage import transform,io

class MyDataset(Dataset):
    """
     Args:
        root (string): 根目录路径
        loader (callable): 根据给定的路径来加载样本的可调用函数
        extensions (list[string]): 可扩展类型列表，即能接受的图像文件类型.
        transform (callable, optional): 用于样本的transform函数，然后返回样本transform后的版本
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): 用于样本标签的transform函数

     Attributes:
        classes (list): 类别名列表
        class_to_idx (dict): 项目(class_name, class_index)字典,如{'cat': 0, 'dog': 1}
        samples (list): (sample path, class_index) 元组列表，即(样本路径, 类别索引)
        targets (list): 在数据集中每张图片的类索引值，为列表
    """

    def __init__(self, root, transform=None, target_transform=None):
        
        samples = make_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.samples = samples
        # self.targets = [s[1] for s in samples]  # 所有图像的类索引值组成的列表

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        
        img_path = self.samples[index]
        mask_path = img_path.split('.')[0] + "_mask."+img_path.split('.')[1]
        
        mask_path = mask_path.replace("images", "masks")

        img = load_img(img_path)
        mask= load_mask(mask_path)
        if self.transform is not None:
            # sample = self.transform(sample)
            sample = {'image':img, 'gt': mask}
            # sample_edge = {'image':img, 'gt': edge_mask}
            sample = self.transform(sample)
            # sample_edge = self.transform(sample_edge)
            img, mask = sample['image'], sample['gt']
                   
        sample = {'image': img, 'gt': mask, 'target': 0}
        return sample

    def __len__(self):
        return len(self.samples)
 
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def extract_number(filename):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def make_dataset(dir):
    """
        返回形如[(图像路径, 该图像对应的类别索引值),(),...]
    """
    images = []
    img_dir = dir + "/images"
    # mask_dir = dir + "/masks"
    img_dir = os.path.expanduser(img_dir)
    # mask_dir = os.path.expanduser(mask_dir)
    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in sorted(fnames, key=extract_number):
                if fname.endswith((".png", ".jpg")) : #查看文件是否是支持的可扩展类型，是则继续
                    image_path = os.path.join(root, fname)
                    
                    # mask_path = os.path.join(mask_dir, fname.split('.')[0]+"_mask."+fname.split('.')[1])
                    images.append(image_path)
    # print(len(images))
    return images

def load_img(path):
    # print(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (512, 512))
    image = image / 255.0
    # image = Image.fromarray(np.uint8(image))
    image = np.array(image, dtype=np.float32)
    return image

def load_mask(path):
  
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   

    mask = mask / 255.0
    
    mask = np.array(mask, dtype=np.float32)
    
    return mask