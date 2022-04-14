import torch
from torch import nn, einsum
from einops import repeat
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils

# for edge downsampling
class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.residual = nn.Conv2d(in_dim, out_dim, 1)
        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(out_dim)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x1 = self.norm1(self.conv(x))
        x2 = self.norm2(self.residual(self.pool(x)))
        x = self.act(x1 + x2)
        return x

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class Transform(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(Transform, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class Attention(nn.Module):
    """ Attention module"""
    def __init__(self, in_channel, nf):
        super(Attention, self).__init__()
        
        #self.softmax = nn.Softmax()
        self.softmax = nn.Softmax(dim=1)
        self.q_conv = nn.Conv2d(in_channel, nf, 1, padding=0, bias=True)
        self.k_conv = nn.Conv2d(in_channel, nf, 1, padding=0, bias=True)
        self.v_conv = nn.Conv2d(in_channel, nf, 1, padding=0, bias=True)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, query, key, x):
        b, c, h, w = x.shape

        query = self.q_conv(query)
        key = self.k_conv(key)
        x = self.v_conv(x)
        #print(query.shape, key.shape, x.shape)

        proj_query = query.view(b, c, -1)
        proj_key = key.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(b, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)
        out = self.drop(out)

        return out

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Attention Branch
class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)


        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class MergeInfo(nn.Module):

    def __init__(self, nf, reduction=2, K=2, t=30):
        super(MergeInfo, self).__init__()
        self.t=t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranch(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)         
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x_main, x_edge, x_sen):
        residual = x_edge
        a, b, c, d = x_edge.shape #

        x = self.conv_first(x_main)
        x = self.lrelu(x)

        # Attention Dropout
        #y = self.avg_pool(x_sen).view(a,b)
        y = self.ADM(x_sen)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out



class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output
        
class UpPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(UpPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)

        
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output

if __name__ == '__main__':
    ca = MergeInfo(64).cuda()
    in_ = torch.ones([6, 64, 256, 256]).cuda()
    out = ca(in_, in_, in_)
    print(out.shape)

    attn = Attention(64, 64).cuda()
    out = attn(in_, in_, in_)
    print(out.shape)