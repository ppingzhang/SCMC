import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np 
from model.utils import Downsample, Transform, UpPixelConvolutionalBlock, MergeInfo, AttentionBlock, ResidualBlock

from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.models.priors import CompressionModel
from compressai.entropy_models import GaussianConditional

from compressai.layers import GDN



# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class InfoExtractor(nn.Module):
    def __init__(self, in_channel=3, nf = 64, device="cuda"):
        super(InfoExtractor, self).__init__()
    
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
        
        f1 = self.down1(self.Prel1(self.conv1(x)))
        f2 = self.down2(self.Prel2(self.conv2(f1)))
        f3 = self.down3(self.Prel3(self.conv3(f2)))
        f4 = self.down4(self.Prel4(self.conv4(f3)))
        
        return {'f1': f1, 
                'f2': f2, 
                'f3': f3, 
                'f4': f4}

class StructureExtractor(nn.Module):
    def __init__(self, in_channel=1, nf = 64, device="cuda"):
        super(StructureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, nf, 3, 1, 1)
        self.Prel1 = nn.PReLU()
        self.down1 = nn.Conv2d(nf, nf, 3, 2, 1)

        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel2 = nn.PReLU()
        self.down2 = nn.Conv2d(nf, nf, 3, 2, 1)

        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.Prel3 = nn.PReLU()
        self.down3 = nn.Conv2d(nf, nf, 3, 2, 1)


    def forward(self, x):
        
        f1 = self.down1(self.Prel1(self.conv1(x)))
        f2 = self.down2(self.Prel2(self.conv2(f1)))
        f3 = self.down3(self.Prel3(self.conv3(f2)))

        return {'f1': f1, 
                'f2': f2, 
                'f3': f3}

class HyperpriorAnalysis(nn.Module):
    """
    Hyperprior 'analysis model' as proposed in [1]. 

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of input channels
    """
    def __init__(self, C=220, N=320, activation='relu'):
        super(HyperpriorAnalysis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.activation = getattr(F, activation)
        self.n_downsampling_layers = 2

        self.conv1 = nn.Conv2d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv2d(N, N, **cnn_kwargs)

    def forward(self, x):
        
        # x = torch.abs(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x

class HyperpriorSynthesis(nn.Module):
    """
    Hyperprior 'synthesis model' as proposed in [1]. Outputs 
    distribution parameters of input latents.

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of output channels
    """
    def __init__(self, C=220, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(N, C, kernel_size=3, stride=1, padding=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.conv3(x2)

        if self.final_activation is not None:
            x3 = self.final_activation(x3)
        return x3

class L3_Codec_Hyperprior(CompressionModel):
    def __init__(self, in_channel=3, nf = 128, device="cuda"):

        N = 128
        M = 192
        super().__init__(entropy_bottleneck_channels=N)
        

        self.device = device
        self.nf = nf
        

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s_d1 = deconv(N, N)
        self.g_s_g1 = GDN(N, inverse=True)
        self.g_s_d2 = deconv(N, N)
        self.g_s_g2 = GDN(N, inverse=True)
        self.g_s_d3 = deconv(N, N)
        self.g_s_g3 = GDN(N, inverse=True)
        self.g_s_d4 = deconv(N, 3)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # senmantic extractor
        self.sobelinfo_extractor = InfoExtractor(in_channel=in_channel, nf = nf)
        self.info_extractor = nn.Conv2d(in_channel, nf, 3, 2, 1)

        self.edge_conv1 = nn.Conv2d(nf*2, nf, 3, 1, 1)
        self.edge_conv2 = nn.Conv2d(nf*2, nf, 3, 1, 1)
        self.edge_conv3 = nn.Conv2d(nf*2, nf, 3, 1, 1)


        self.conv0 = nn.Conv2d(M, nf, 1, 1, 0)

        self.down1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.convy = nn.Conv2d(nf, N, 3, 1, 1)

        # structure extractor
        self.edge_extractor = StructureExtractor(in_channel=1, nf = nf)

        self.tran1 = Transform(nf)
        self.tran2 = Transform(nf)
        self.tran3 = Transform(nf)

        self.m1 = MergeInfo(nf)
        self.m2 = MergeInfo(nf)
        self.m3 = MergeInfo(nf)

        self.gaussian_conditional = GaussianConditional(None)

        self.mse = nn.MSELoss()
    
    def edge_conv2d(self, im, device='cuda'):
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

        conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device)

        edge_detect = conv_op(im)
        return edge_detect

    def forward(self, im_map, base_map, edge_map):

        #y_hat, y_likelihoods = self.entropy_bottleneck(y)

        im_map = im_map.float()
        edge_map = edge_map.float()
        base_map = base_map.float()

        y = self.g_a(im_map)
        z = self.h_a(torch.abs(y))
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
 

        sobel_edge = self.edge_conv2d(base_map)
        sobel_feats = self.sobelinfo_extractor(sobel_edge)
        edge_feats = self.edge_extractor(edge_map)

        #Q:edge
        #K:res
        #V:res/previous
        
        edge1 = self.edge_conv1(torch.cat([edge_feats['f1'], sobel_feats['f2']], 1))
        edge2 = self.edge_conv2(torch.cat([edge_feats['f2'], sobel_feats['f3']], 1))
        edge3 = self.edge_conv3(torch.cat([edge_feats['f3'], sobel_feats['f4']], 1))

        
        t1 = self.tran1(edge3)
        t2 = self.tran2(edge2)
        t3 = self.tran3(edge1)

        y_hat = self.conv0(y_hat)
        fm1 = self.m1(y_hat, t1, y_hat)
        fm1 = self.g_s_d1(fm1)
        fm1 = self.g_s_g1(fm1)

        fm2 = self.m2(fm1, t2, fm1)
        fm2 = self.g_s_d2(fm2)
        fm2 = self.g_s_g2(fm2)

        fm3 = self.m3(fm2, t3, fm2)
        fm3 = self.g_s_d3(fm3)
        fm3 = self.g_s_g3(fm3)


        x_hat = self.g_s_d4(fm3)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods, "z": z_likelihoods
            },
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def compress(self, im_map, base_map, edge_map):
        im_map = im_map.float()
        edge_map = edge_map.float()
        base_map = base_map.float()
        

        y = self.g_a(im_map)
        z = self.h_a(torch.abs(y))
        

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
        
    def decompress(self, strings, shape, edge_map, base_map):
        assert isinstance(strings, list) and len(strings) == 2
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)

        edge_map = edge_map.float()
        base_map = base_map.float()
        
        
        sobel_edge = self.edge_conv2d(base_map)
        sobel_feats = self.sobelinfo_extractor(sobel_edge)
        edge_feats = self.edge_extractor(edge_map)

        #Q:edge
        #K:res
        #V:res/previous
        
        edge1 = self.edge_conv1(torch.cat([edge_feats['f1'], sobel_feats['f2']], 1))
        edge2 = self.edge_conv2(torch.cat([edge_feats['f2'], sobel_feats['f3']], 1))
        edge3 = self.edge_conv3(torch.cat([edge_feats['f3'], sobel_feats['f4']], 1))

        t1 = self.tran1(edge3)
        t2 = self.tran2(edge2)
        t3 = self.tran3(edge1)

        y_hat = self.conv0(y_hat)
        fm1 = self.m1(y_hat, t1, y_hat)
        fm1 = self.g_s_d1(fm1)
        fm1 = self.g_s_g1(fm1)

        fm2 = self.m2(fm1, t2, fm1)
        fm2 = self.g_s_d2(fm2)
        fm2 = self.g_s_g2(fm2)

        fm3 = self.m3(fm2, t3, fm2)
        fm3 = self.g_s_d3(fm3)
        fm3 = self.g_s_g3(fm3)

        x_hat = self.g_s_d4(fm3)
        return {"x_hat": x_hat}
        
    def loss(self, output, target, lmbda=1):
        out = {}

        out["re_loss"] = self.mse(output["x_hat"], target)* 5 * (2** lmbda)

        ss = output["x_hat"].shape
        num_pixels = ss[0]* ss[2] *ss[3]
        out["bpp_y"] = torch.log(output["likelihoods"]['y']).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]['z']).sum() / (-math.log(2) * num_pixels)

        out["bpp_loss"] = out["bpp_y"] + out["bpp_z"]
        
        out["loss"] = out["re_loss"] + out["bpp_loss"] 

        return out



