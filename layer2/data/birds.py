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

    def __init__(self, crop_size=256, normalize=True, img_file = "", \
                imgs_path = "", edge_path = "",caps_path = "", \
                style_path = '', seed=0, extract=True, training=True):
        
        self.imgs_path = imgs_path
        self.caps_path = caps_path
        self.style_path = style_path
        self.edge_path = edge_path
        


        with open(img_file, 'rb') as f:
            self.imgs_path_list = pickle.load(f, encoding='bytes')

        if training:
            random.seed(seed)
            random.shuffle(self.imgs_path_list)


        self.crop_size = crop_size
        self.normalize = normalize
        self.extract = extract


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

        
        img_name = self.imgs_path_list[idx]

        name = img_name.split('/')[-1]
        img_path = self.imgs_path + name+'.jpg'
        text_path = self.caps_path +'/' +img_name +'.txt'
        g_img_path = self.style_path +'/' +name +'.png'

        #print(img_path, text_path, g_img_path)

        filesize = os.path.getsize(img_path)
        image_name = os.path.basename(img_path)

        
        sobel_name = os.path.basename(img_path)
        sobel_name = self.edge_path + sobel_name
        sobel_name = sobel_name.replace('.jpg', '.png')
        edge_im = np.array(Image.open(sobel_name))/255.0
    
        
        img = Image.open(img_path)
        img = img.convert('RGB') 

        g_img = Image.open(g_img_path)
        g_img = g_img.convert('RGB') 


        W, H = img.size 
        bpp = filesize * 8. / (H * W)
        
        #text_path = '/'.join([self.caps_path, os.path.basename(img_path).split(".")[0] + '.txt'])
        
        with open(text_path) as f:
            line = f.readlines()[0]
            caps = line.replace('\n', '')
        
        dynamic_transform = self._transforms()
        transformed = dynamic_transform(img)
        transformed_g_img = dynamic_transform(g_img)

        transformed_edge = torch.unsqueeze(torch.tensor(edge_im), 0)

        return transformed, bpp, caps, image_name, transformed_edge, transformed_g_img

    def __len__(self):
        return len(self.imgs_path_list)

    