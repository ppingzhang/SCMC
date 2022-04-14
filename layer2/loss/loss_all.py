
import argparse
import math

import torch
import torch.nn as nn
from loss.Vgg19 import VGG19_LossNetwork
from loss.sobel_loss import SobelLoss
from loss.FFTLoss import FFTLoss

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
        


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
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


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        self.lmbda = self.lmbda_list[lmbda]


    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        tt = 0
        
        for nn in output["likelihoods"]:
            #print(nn)
            out[nn] = torch.log(output["likelihoods"][nn]).sum() / (-math.log(2) * num_pixels)

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class RateDistortionLossL1(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        self.lmbda = self.lmbda_list[lmbda]


    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        tt = 0
        
        for nn in output["likelihoods"]:
            out[nn] = torch.log(output["likelihoods"][nn]).sum() / (-math.log(2) * num_pixels)

        out["im_loss"] = self.lmbda * 255 ** 2 * self.l1(output["x_hat"], target)
        out["loss"] = out["im_loss"] + out["bpp_loss"]*50

        return out



class RateDistortionLossSketch(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        self.lmbda = self.lmbda_list[lmbda]


    def forward(self, output, target_im, output_edge):
        N, _, H, W = target_im.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        tt = 0
        
        for nn in output["likelihoods"]:
            #print(nn)
            out[nn] = torch.log(output["likelihoods"][nn]).sum() / (-math.log(2) * num_pixels)

        out["l_img_loss"] = self.mse(output["x_hat"], target_im)
        out["l_edge_loss"] = self.mse(output["edge_hat"], output_edge)
        out["loss"] = self.lmbda * 255 ** 2 * (out["l_img_loss"] + out["l_edge_loss"]) + out["bpp_loss"]

        return out

class RateDistortionLoss_multi(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda_list[lmbda] * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out



class DistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, target, scale_list):
        out = {}
        out["loss"] = 0.0
        for idx, (img, scale) in enumerate(zip(output, scale_list)):
            out["mse_loss_" + str(scale)] = self.mse(img, target) * scale * 100
            out["loss"] = out["loss"] + out["mse_loss_" + str(scale)] 

            vgg = VGG19_LossNetwork()
            vgg_loss, _ = vgg(img, target)
            out["vgg_" + str(scale)] = vgg_loss * 10
            out["loss"] = out["loss"] + out["vgg_" + str(scale)]

        return out




class DistortionLossMulti(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.vgg = VGG19_LossNetwork()
        self.sobel_ex = SobelLoss()

        self.fft = FFTLoss()
        
        
    def forward(self, output, target):
        out = {}
        
        gt_16 = torch.nn.functional.interpolate(target, size=[16, 16], mode='nearest', align_corners=None)
        gt_64 = torch.nn.functional.interpolate(target, size=[64, 64], mode='nearest', align_corners=None)

        #out["mse_loss_16"] = self.mse(output[0], gt_16) * 5
        #out["mse_loss_64"] = self.mse(output[1], gt_64) * 10
        #out["mse_loss_256"] = self.mse(output[2], target) * 30

        #print(out["mse_loss_256"].shape, target.shape)
        #vgg_loss, _ = self.vgg(output[2], target)
        #out["vgg"] = vgg_loss * 10

        
        #out["loss"] =  out["mse_loss_16"] + out["mse_loss_64"] + out["mse_loss_256"] + out["vgg"]
        #out["loss"] =  out["mse_loss_256"] + out["vgg"]


        #out["mse_loss_16"] = self.mse(output[0], gt_16) * 5
        #out["mse_loss_64"] = self.mse(output[1], gt_64) * 10

        #fft
        '''
        out["mse_loss_256"] = self.mse(output[2], target) * 10

        vgg_loss, _ = self.vgg(output[2], target)
        out["vgg"] = vgg_loss * 5
        
        out['sobel'] = self.l1(self.sobel_ex(output[2]), self.sobel_ex(target))*0.05

        out['fft'] = self.l1(self.fft(output[2]), self.fft(target))*0.005
        #out["loss"] =  out["mse_loss_256"] + out["vgg"] + out['sobel'] + out['fft']
        out["loss"] =  out["mse_loss_256"] + out["vgg"] 
        '''

        out["mse_loss_256"] = self.mse(output[2], target)*10

        vgg_loss, _ = self.vgg(output[2], target)
        out["vgg"] = vgg_loss * 50
        
        out['sobel'] = self.l1(self.sobel_ex(output[2]), self.sobel_ex(target))*0.1

        #out['fft'] = self.l1(self.fft(output[2]), self.fft(target))*0.005
        #out["loss"] =  out["mse_loss_256"] + out["vgg"] + out['sobel'] + out['fft']
        out["loss"] =  out["mse_loss_256"] + out["vgg"] + out['sobel']

        return out



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
