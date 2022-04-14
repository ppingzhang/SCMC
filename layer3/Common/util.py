

import os
import math
import shutil

import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image


def save_images(save_img_list, fname, nrow):

    imgs = torch.cat(save_img_list, dim=0)
    save_image(imgs, fname, nrow, normalize=True, scale_each=True)



def save_checkpoint(state, save_path, filename="checkpoint.pth.tar", is_best=False):
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	torch.save(state, save_path+filename)
	if is_best:
		shutil.copyfile(save_path+filename, save_path+"/best.pth.tar")

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
	mse = F.mse_loss(a, b).item()
	return -10 * math.log10(mse)

def tensor2img(tensor_im, save_im):
	unloader = transforms.ToPILImage()
	image = tensor_im.cpu().clone()  
	image = image.squeeze(0) 
	image = torch.clamp(image, 0, 1)
	image = unloader(image)

	dpath = os.path.dirname(save_im) 
	
	if not os.path.exists(dpath):
		os.makedirs(dpath)
		
	image.save(save_im)


def print_loss(output):
	str_p = ""
	for k in output:
		str_p += f'\t{k}: {output[k].item():.5f} |'
	return str_p

			

def show_in_board(writer, step, **imgs):


	for k in imgs:

		in_clamp = torch.clamp(imgs[k], 0, 1)
		in_grid = torchvision.utils.make_grid(in_clamp)
		writer.add_image(k, in_grid, step)



def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
