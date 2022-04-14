import os
import glob 

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from PIL import Image
import math 

import torch
from PIL import Image

import random
import cv2
import numpy as np 

import pickle


class Birds_Edge(Dataset):

    def __init__(self, crop_size=256, normalize=True, img_file="", imgs_path = '', edge_path='', base_img_path='', seed=0, training=True):
        
        self.imgs_path = imgs_path
        self.base_img_path = base_img_path
        self.edge_path = edge_path
        

        self.imgs = []

        with open(img_file, 'rb') as f:
            imgs_path = pickle.load(f, encoding='bytes')


        for img_path in imgs_path:
            img_name = img_path.split('/')[-1]
            self.imgs.append(self.imgs_path + img_name+'.jpg')
    
        if training:
            random.seed(seed)
            random.shuffle(self.imgs)


        if len(self.imgs) == 0:
            raise ValueError(f"There is no image in this path: {self.imgs_path}:")

        self.crop_size = crop_size
        self.normalize = normalize


    def _transforms(self):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [transforms.RandomCrop(self.crop_size),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)


    def __getitem__(self, idx):

        img_path = self.imgs[idx]
        
        filesize = os.path.getsize(img_path)
        image_name = os.path.basename(img_path)

        base_img_path = self.base_img_path+'/' +image_name

        
        edge_name = os.path.basename(img_path)
        edge_name = self.edge_path + edge_name
        edge_name = edge_name.replace('.jpg', '.png')
        edge_im = np.array(Image.open(edge_name))/255.0
    
        
        img = Image.open(img_path)
        img = img.convert('RGB') 

        base_img = Image.open(base_img_path)
        base_img = base_img.convert('RGB') 


        W, H = img.size 
        bpp = filesize * 8. / (H * W)

        
        dynamic_transform = self._transforms()
        transformed = dynamic_transform(img)
        transformed_base_img = dynamic_transform(base_img)

        transformed_edge = torch.unsqueeze(torch.tensor(edge_im), 0)

        return transformed, bpp, image_name, transformed_edge, transformed_base_img

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return tuple(self.imgs.size())

