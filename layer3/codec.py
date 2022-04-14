
import os
import random
from PIL import Image
import time
import numpy as np
import math
import logging
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim


from tensorboardX import SummaryWriter
from torchvision import transforms
from pytorch_msssim import ms_ssim
import torch.nn as nn
from Common.util import save_checkpoint, psnr, tensor2img, print_loss, show_in_board, pad, crop

import config as config

from Common.write_bin import read_from_bin, write_to_bin
from model import create_model
from data.dataset_load import load_test_dataset

torch.backends.cudnn.deterministic = True 



def encode(args, x, ckpt_path, save_bin_path, lmbda):

    en_start = time.time()
    net = create_model(args).cuda()
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    x = x.unsqueeze(0).cuda()
    h, w = x.size(2), x.size(3)
    num_pixels = x.size(1) * h * w
    x_padded = pad(x, p=2 ** 6)

    out_enc = net.compress(x_padded)

    bpp = write_to_bin(out_enc, h=h, w=w, save_path=save_bin_path, lmbda=lmbda)
    enc_time = time.time() - en_start
    bits_bin = os.path.getsize(save_bin_path)
    bits_bin = bits_bin * 8
    bpp = bits_bin / num_pixels
    return bpp, enc_time



def decode(args, save_bin_path, save_img_name):

    de_start = time.time()
    net = create_model(args).cuda()
    net.eval()
    
    #x = x.unsqueeze(0).cuda()
    #h, w = x.size(2), x.size(3)
    strings, original_size, shape, lmbda = read_from_bin(save_bin_path)

    out_dec = net.decompress(strings, shape)
    dec_time = time.time() - de_start

    out_dec["x_hat"] = crop(out_dec["im_x_hat"], (shape[0], shape[1]))
    tensor2img(out_dec["x_hat"], save_img_name)

    return out_dec["x_hat"], dec_time

def test_all(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    lmbda_list = [1, 5, 20, 50, 100]
    result_array = np.zeros([5, 8])

    for jj in [0, 1, 2, 3]:  # range(0, 7): # range(0, 7):
        args.lmbda = lmbda_list[jj]
        net = create_model(args)
        net.cuda()

        for kk in range(0, 1):
            if lmbda_list[jj] == 1:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=40)
            elif lmbda_list[jj] == 5:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=49)
            elif lmbda_list[jj] == 20:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=45) #54
            elif lmbda_list[jj] == 50:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=47) #54

            transform = transforms.Compose([transforms.ToTensor()])
            
            img_list = load_test_dataset('clic')
            
            psnr_all = []
            ssim_all = []
            bpp_all = []
            enc_time_all = []
            dec_time_all = []

            for img_path in img_list:

                img = Image.open(img_path).convert('RGB')
                save_img_name = img_path.replace(
                    '/test_dataset/',
                    '/result_new/Ours/{model}/{label_str}/{lmbda}/'.format(
                        model=args.model,
                        label_str=args.label_str,
                        lmbda=lmbda_list[jj]))
                save_bin_path = img_path.replace(
                    '/test_dataset/',
                    '/result_new/Ours_bin/{model}/{label_str}/{lmbda}/'.format(
                        model=args.model,
                        label_str=args.label_str,
                        lmbda=lmbda_list[jj]))
                
                if save_img_name == img_path:
                    raise ValueError(f"save_name:{save_img_name} == img_path:{img_path}")
                
                bpp, enc_time = encode(args, img, ckpt_path, save_bin_path, lmbda_list[jj])
                de_img, dec_time = decode(args, save_bin_path, save_img_name)

                psnr_value = psnr(img, de_img)
                ssim_value = ms_ssim(img, de_img)
                psnr_all.append(psnr_value)
                ssim_all.append(ssim_value)
                bpp_all.append(bpp)
                enc_time_all.append(enc_time)
                dec_time_all.append(dec_time)

                result_str1 = '{}-----psnr:{:.2f}, ssim:{:.2f}, bpp:{:.2f}, enc_time:{:.2f}, dec_time:{:.2f}'.format(
                    args.model,
                    psnr_value,
                    ssim_value,
                    bpp,
                    enc_time,
                    dec_time)
                #print(result_str1)

            psnr_all = np.array(psnr_all)
            ssim_all = np.array(ssim_all)
            bpp_all = np.array(bpp_all)
            enc_time_all = np.array(enc_time_all)
            dec_time_all = np.array(dec_time_all)
            # print(bpp_all)
            result_array[0, jj] = np.mean(psnr_all)
            result_array[1, jj] = np.mean(ssim_all)
            result_array[2, jj] = np.mean(bpp_all)
            result_array[3, jj] = np.mean(enc_time_all)
            result_array[4, jj] = np.mean(dec_time_all)

            result_str = '{}-----psnr:{:.2f}, ssim:{:.2f}, bpp:{:.2f}, enc_time:{:.2f}, dec_time:{:.2f}'.format(
                args.model,
                np.mean(psnr_all),
                np.mean(ssim_all),
                np.mean(bpp_all),
                np.mean(enc_time_all),
                np.mean(dec_time_all))
            print(result_str)

        print('----------result------')
        for ii in range(5):
            print(result_array[ii, :])
            print("\n")

        print('----------result------')
        for ii in range(5):
            result_str = ''
            for kk in range(8):
                result_str += f'{result_array[ii, kk]}\t'
            print(result_str)
            print("\n")

def test_single_img(model, x, p_x, save_name, lmbda):
    model.eval()
    x = x.unsqueeze(0)
    x = x.cuda()

    h, w = x.size(2), x.size(3)
    x_padded = pad(x, p=2 ** 6)

    en_start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - en_start

    save_path = './out.bin'
    bpp1 = write_to_bin(out_enc, h=h, w=w, save_path=save_path, lmbda=lmbda)
    bits_bin = os.path.getsize(save_path)
    bits_bin = bits_bin * 8
    
    pixel_bits = bits_bin / num_pixels
    strings, original_size, shape, lmbda = read_from_bin(save_path)

    de_start = time.time()
    out_dec = model.decompress(strings, shape)
    dec_time = time.time() - de_start

    out_dec["im_x_hat"] = crop(out_dec["im_x_hat"], (h, w))
    tensor2img(out_dec["im_x_hat"], save_name)

    out_dec_forward = model(x_padded, x_padded, False)
    out_dec_forward["im_x_hat"] = crop(out_dec_forward["im_x_hat"], (h, w))
    
    bpp_forward = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_dec_forward["likelihoods"].values()
    )

    bpp_forward = bpp_forward.cpu().numpy()

    return {
        "psnr_forward": psnr(p_x, out_dec_forward["im_x_hat"]),
        "psnr_real": psnr(p_x, out_dec["im_x_hat"]),
        "ms_ssim_real": ms_ssim(p_x, out_dec["im_x_hat"], data_range=1.0).item(),
        "bpp_real": pixel_bits,
        "bpp_theroy": bpp_forward,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

def test_ReNoIR_(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    lmbda_list = [1, 5, 20, 50, 100]
    result_array = np.zeros([5, 8])

    for jj in [0, 1, 2, 3]:  # range(0, 7): # range(0, 7):
        args.lmbda = lmbda_list[jj]
        net = create_model(args)
        net.cuda()

        #ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
        #    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=49)
        for kk in range(69, 70):
            if lmbda_list[jj] == 1:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=40)
            elif lmbda_list[jj] == 5:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=49)
            elif lmbda_list[jj] == 20:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=45) #54
            elif lmbda_list[jj] == 50:
                ckpt_path = './ckpt/{model}/{label_str}/{lmbda}/00{kk}.pth.tar'.format(
                    model=args.model, label_str=args.label_str, lmbda=lmbda_list[jj], kk=47) #54

            checkpoint = torch.load(ckpt_path)
            net.update(force=True)
            print(ckpt_path, checkpoint['epoch'])
            net.load_state_dict(checkpoint['state_dict'])
            for parameter in net.parameters():
                parameter.requires_grad = False

            net.eval()
            image_path = []

            transform = transforms.Compose([transforms.ToTensor()])
            #print(args.test_dataset)
            gt_img_list, de_img_list = ReNoIR_1024_data_load()
            #print(gt_img_list)
            psnr_all = []
            psnr_post_all = []
            ssim_all = []
            bpp_all = []
            bpp_bin_all = []
            enc_time_all = []
            dec_time_all = []

            for img_path, p_img_path in zip(de_img_list, gt_img_list):

                img = Image.open(img_path).convert('RGB')
                image_or = Image.open(p_img_path).convert('RGB')
                save_name = img_path.replace(
                    '/test_dataset/',
                    '/result_new/Ours/{model}/{label_str}/{lmbda}/'.format(
                        model=args.model,
                        label_str=args.label_str,
                        lmbda=lmbda_list[jj]))
                # rint(save_name)
                if save_name == img_path:
                    raise ValueError(f"save_name:{save_name} == img_path:{img_path}")
                result = test_single_img(
                    net,
                    transform(img),
                    transform(image_or),
                    save_name, lmbda_list[jj])
                psnr_all.append(result['psnr'])
                psnr_post_all.append(result['psnr_post'])
                ssim_all.append(result['ms_ssim'])
                bpp_all.append(result['bpp_theroy'])
                bpp_bin_all.append(result['bpp_real'])
                enc_time_all.append(result['encoding_time'])
                dec_time_all.append(result['decoding_time'])

                result_str1 = '{}-----psnr_real:{:.2f}, psnr_theory:{:.2f}, ssim:{:.2f}, bpp_theory:{:.2f}, bpp_real:{:.2f}, enc_time:{:.2f}, dec_time:{:.2f}'.format(
                    args.model,
                    result['psnr'],
                    result['psnr_post'],
                    result['ms_ssim'],
                    result['bpp_theroy'],
                    result['bpp_real'],
                    result['encoding_time'],
                    result['decoding_time'])
                #print(result_str1)

            psnr_all = np.array(psnr_all)
            psnr_post_all = np.array(psnr_post_all)
            ssim_all = np.array(ssim_all)
            bpp_all = np.array(bpp_all)
            bpp_bin_all = np.array(bpp_bin_all)
            enc_time_all = np.array(enc_time_all)
            dec_time_all = np.array(dec_time_all)
            # print(bpp_all)
            result_array[0, jj] = np.mean(psnr_all)
            result_array[1, jj] = np.mean(psnr_post_all)
            result_array[2, jj] = np.mean(ssim_all)
            result_array[3, jj] = np.mean(bpp_all)
            result_array[4, jj] = np.mean(bpp_bin_all)

            result_str = '{}-----psnr_real:{:.2f}, psnr_theory:{:.2f}, ssim:{:.2f}, bpp_theory:{:.2f}, bpp_real:{:.2f}, enc_time:{:.2f}, dec_time:{:.2f}'.format(
                args.model,
                np.mean(psnr_all),
                np.mean(psnr_post_all),
                np.mean(ssim_all),
                np.mean(bpp_all),
                np.mean(bpp_bin_all),
                np.mean(enc_time_all),
                np.mean(dec_time_all))
            print(result_str)

        print('----------result------')
        for ii in range(5):
            print(result_array[ii, :])
            print("\n")

        print('----------result------')
        for ii in range(5):
            result_str = ''
            for kk in range(8):
                result_str += f'{result_array[ii, kk]}\t'
            print(result_str)
            print("\n")



if __name__ == "__main__":
    args = config.args

    if args.mode == 'test':
        test_all(args)
    