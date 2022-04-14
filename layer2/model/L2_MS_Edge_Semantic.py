'''
Multi scale edge and semantic 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np 

from model.utils import Downsample, Transform, UpPixelConvolutionalBlock, MergeInfo, Attention, ResidualBlock

import math

from DISTS_pytorch import DISTS
from IQA_pytorch import SSIM

class SemanticExtractor(nn.Module):
    def __init__(self, in_channel=3, nf = 64, device="cuda"):
        super(SemanticExtractor, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channel, nf, 3, 1, 1)
        self.Prel1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(5, 2, 2)

        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(5, 2, 2)

        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(3, 2, 1)

        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel4 = nn.PReLU()
        self.pool4 = nn.MaxPool2d(3, 2, 1)

        self.conv5 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel5 = nn.PReLU()
        self.pool5 = nn.MaxPool2d(3, 2, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.fc1 = nn.Linear(nf, nf, bias=False)
        self.pr1 = nn.PReLU()

        self.fc2 = nn.Linear(nf, nf, bias=False)
        self.fc3 = nn.Linear(nf, nf, bias=False)
        self.fc4 = nn.Linear(nf, nf, bias=False)

    def forward(self, x):

        fm_s1 = self.pool2(self.Prel2(self.pool1(self.Prel1(self.conv1(x)))))
        fm_s2 = self.pool3(self.Prel3(self.conv3(fm_s1)))
        fm_s3 = self.pool4(self.Prel4(self.conv4(fm_s2)))
        fm_s4 = self.pool5(self.Prel5(self.conv5(fm_s3)))

        B, C, _, _ = fm_s4.shape
        avg_f = self.avg_pool(fm_s4).reshape([B, C])

        fc_1 = self.fc2(self.pr1(self.fc1(avg_f)))
        fc_2 = self.fc3(fc_1)
        fc_3 = self.fc4(fc_2)


        return {'f1': fm_s1, 
                'f2': fm_s2, 
                'f3': fm_s3, 
                'f4': fm_s4, 
                'fc1': fc_1,
                'fc2': fc_2,
                'fc3': fc_3}

class StructureExtractor(nn.Module):
    def __init__(self, in_channel=1, nf = 64, device="cuda"):
        super(StructureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, nf, 3, 1, 1)
        self.Prel1 = nn.PReLU()
        self.down1 = Downsample(nf, nf)

        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel2 = nn.PReLU()
        self.down2 = Downsample(nf, nf)

        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel3 = nn.PReLU()
        self.down3 = Downsample(nf, nf)

        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel4 = nn.PReLU()
        self.down4 = Downsample(nf, nf)

    def forward(self, x):
        fm_d1 = self.down1(self.Prel1(self.conv1(x)))
        fm_d2 = self.down2(self.Prel2(self.conv2(fm_d1)))
        fm_d3 = self.down3(self.Prel3(self.conv3(fm_d2)))
        fm_d4 = self.down4(self.Prel4(self.conv4(fm_d3)))

        return {'f1': fm_d1,
                'f2': fm_d2,
                'f3': fm_d3,
                'f4': fm_d4}



def attn_block():
    pass

class L2_MS_Edge_Semantic(nn.Module):
    def __init__(self, in_channel=3, nf = 64, device="cuda"):
        super(L2_MS_Edge_Semantic, self).__init__()
        
        
        self.device = device
        self.nf = nf
        
        # senmantic extractor
        self.semantic_extractor = SemanticExtractor(in_channel=in_channel, nf = nf)

        self.attn1 = Attention(nf, nf)
        self.attn2 = Attention(nf, nf)
        self.attn3 = Attention(nf, nf)
        self.attn4 = Attention(nf, nf)


        self.down1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down4 = nn.Conv2d(nf, nf, 3, 2, 1)

        # structure extractor
        self.structure_extractor = StructureExtractor(in_channel=1, nf = nf)

        self.tran1 = Transform(nf)
        self.tran2 = Transform(nf)
        self.tran3 = Transform(nf)
        self.tran4 = Transform(nf)

        self.m1 = MergeInfo(nf)
        self.m2 = MergeInfo(nf)
        self.m3 = MergeInfo(nf)
        self.m4 = MergeInfo(nf)
  
        self.up1 = UpPixelConvolutionalBlock(kernel_size=3, n_channels=nf, scaling_factor=2)
        self.up2 = UpPixelConvolutionalBlock(kernel_size=3, n_channels=nf, scaling_factor=2)
        self.up3 = UpPixelConvolutionalBlock(kernel_size=3, n_channels=nf, scaling_factor=2)

        self.up4 = UpPixelConvolutionalBlock(kernel_size=3, n_channels=nf, scaling_factor=2)
        self.rs4 = ResidualBlock(kernel_size=3, n_channels=nf)
        self.up5 = UpPixelConvolutionalBlock(kernel_size=3, n_channels=nf, scaling_factor=2)
        self.rs5 = ResidualBlock(kernel_size=3, n_channels=nf)

        self.re = nn.Conv2d(nf, 3, 3, 1, 1)

        self.l1 = nn.L1Loss()
        self.dists = DISTS()


    def forward(self, structure_map, semantic_map):

        structure_map = structure_map.float()
        semantic_map = semantic_map.float()
        
        if not structure_map.shape[-1] == 128:
            structure_map = torch.nn.functional.interpolate(structure_map, size=[128, 128], mode='nearest', align_corners=None)
        
        #imgs_feats = self.encoder(edge)
        # f1 [64, 64] 
        # f2 [32, 32]
        # f3 [16, 16]
        # f4 [8,  8 ]
        # f1, f2, f3, f4, fc1, fc2, fc3
        semantic_out = self.semantic_extractor(semantic_map)
        structure_out = self.structure_extractor(structure_map)

        #Q:semantic
        #K:stucture
        #V:structure/previous one
        a1 = self.attn1(semantic_out['f1'], structure_out['f1'], structure_out['f1'])
        a2 = self.attn2(semantic_out['f2'], structure_out['f2'], a1)
        a2 = self.down2(a2)
        a3 = self.attn3(semantic_out['f3'], structure_out['f3'], a2)
        a3 = self.down3(a3)
        a4 = self.attn4(semantic_out['f4'], structure_out['f4'], a3)
        a4 = self.down4(a4)

        t1 = self.tran1(structure_out["f1"])
        t2 = self.tran2(structure_out["f2"])
        t3 = self.tran3(structure_out["f3"])
        t4 = self.tran4(structure_out["f4"])

        
        fu = a4 + t4
        fm1 = self.up1(fu)
        fm1 = self.m1(fm1, t3, semantic_out['fc1'])#forward(self, x_main, x_edge, x_sen)
        
        fm2 = self.up2(fm1)
        fm2 = self.m2(fm2, t2, semantic_out['fc2'])

        fm3 = self.up3(fm2)
        fm3 = self.m3(fm3, t1, semantic_out['fc3'])
        

        fm4 = self.rs4(self.up4(fm3))
        fm5 = self.rs5(self.up5(fm4))
        de_img = self.re(fm5)

        return {"x_hat":de_img}
    
    def loss(self, output, target):
        out = {}
        
        out["re_loss"] = self.l1(output["x_hat"], target)*10

        out['dists'] = self.dists(output["x_hat"], target, require_grad=True, batch_average=True) *10

        out["loss"] =  out["re_loss"] + out['dists'] 

        return out
        
if __name__=='__main__':
    ml = SemanticExtractor().cuda()
    img = torch.zeros([4, 3, 256, 256]).cuda()
    de_img_list = ml(img)
    print(de_img_list["fm_s1"].shape, de_img_list["fm_s5"].shape, de_img_list["fc_1"].shape, de_img_list["fc_2"].shape, de_img_list["fc_3"].shape)