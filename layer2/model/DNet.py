
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, image_dims, spectral_norm=True):
        """ 
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()
        
        self.image_dims = image_dims
        print(self.image_dims, '=======')
        im_channels = self.image_dims[0]
        kernel_dim = 4
        filters = (64, 128, 256, 512)


        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        
        self.conv1 = norm(nn.Conv2d(im_channels, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """
        
        x2 = self.activation(self.conv1(x))
        x3 = self.activation(self.conv2(x2))
        x4 = self.activation(self.conv3(x3))
        x5 = self.activation(self.conv4(x4))
        
        out_logits = self.conv_out(x5).view(-1,1)
        out = torch.sigmoid(out_logits)
        
        return out, out_logits



class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

if __name__ == "__main__":
    B = 10
    C = 128
    S = 4
    print('Image 1')
    x_gen = torch.randn((B,3,256,256))
    x_real = torch.randn((B,3,256,256))
    latents = torch.randn((B,C,S,S))

    
    Disc_out = Discriminator(image_dims=[3,256,256])
    print('Discriminator output', x_gen.size())


    D_out, D_out_logits = Disc_out(x_real)
    D_out = torch.squeeze(D_out)
    D_out_logits = torch.squeeze(D_out_logits)

    D_real, D_gen = torch.chunk(D_out, 2, dim=0)
    D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

    print(D_out.shape, D_out_logits.shape)
    print(D_real.shape, D_gen.shape)
    print(D_real_logits.shape, D_gen_logits.shape)