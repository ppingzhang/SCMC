
import argparse
import math

import torch
import torch.nn as nn

import torch.nn.functional as F



class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        





class DistortionLossEdge(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, target, edges, edge_gt, scale_list):
        out = {}
        out["loss"] = 0.0
        for idx, (img, edge, scale) in enumerate(zip(output, edges, scale_list)):
            out["mse_loss_" + str(scale)] = self.mse(img, target) * scale * 100
            out["l1_loss_" + str(scale)] = self.l1(edge_gt, edge) * scale * 10

            out["loss"] = out["loss"] + out["mse_loss_" + str(scale)]  + out["l1_loss_" + str(scale)]

            vgg = VGG19_LossNetwork()
            vgg_loss, _ = vgg(img, target)
            out["vgg_" + str(scale)] = vgg_loss * 10
            out["loss"] = out["loss"] + out["vgg_" + str(scale)]

        return out
