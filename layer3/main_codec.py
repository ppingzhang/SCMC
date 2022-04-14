'''
layer 2 for the basic map
'''


import os
import random
from PIL import Image
import time
import numpy as np
import math
import logging
import torch.optim as optim
import clip

from tqdm import tqdm
import cv2


import torch
from tensorboardX import SummaryWriter
from Common.util import save_checkpoint, print_loss

import config as config
import importlib
from loss.loss_all import AverageMeter

from Common.write_bin import read_from_bin, write_to_bin
from data.dataloader import DataLoader_Edge_All 

from torchvision import transforms
from DISTS_pytorch import DISTS
from lpips_pytorch import LPIPS


from Common.util import save_images
from PIL import Image

Tensor = torch.cuda.FloatTensor
torch.backends.cudnn.deterministic = True

pkg = importlib.import_module('model.'+config.args.model)
Model =  getattr(pkg, config.args.model)


logger = logging.getLogger(config.args.model)
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')

stdhandler = logging.StreamHandler()
stdhandler.setFormatter(formatter)
logger.addHandler(stdhandler)

if not os.path.exists(f'../ckpt/layer3/log/{config.args.lmbda}/'):
    os.makedirs(f'../ckpt/layer3/log/{config.args.lmbda}/')

filehandler = logging.FileHandler(f'../ckpt/layer3/log/{config.args.model}_{config.args.label_str}_{config.args.lmbda}.log')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

step = 0
writer = SummaryWriter(f'../ckpt/layer3/log/{config.args.model}/{config.args.label_str}/{config.args.lmbda}/')
save_img_path = f"../ckpt/layer3/figures/{config.args.model}/{config.args.label_str}/{config.args.lmbda}/"

if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
        
def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    #assert len(inter_params) == 0
    #print(params_dict.keys()-union_params)
    #assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def train_one_epoch(args, model, train_dataloader, optimizer, aux_optimizer, epoch, save_img_path):
    global step
    model.train()
    device = next(model.parameters()).device
    
    for i, (p_im, _, _, edge_im, base_im) in enumerate(train_dataloader):
        #print(edge_im.shape, style_im.shape)
        p_im = p_im.to(device)
        edge_im = edge_im.to(device)
        base_im = base_im.to(device)
        #edge_im = torch.nn.functional.interpolate(edge_im, size=[128, 128], mode='nearest', align_corners=None)
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
       
        out_net = model(p_im, base_im, edge_im)
        out_criterion = model.loss(out_net, p_im, args.lmbda)

        out_criterion["loss"].backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        step = step + 1

        out_criterion['aux_loss'] = aux_loss

        if i % args.print_interval == 0:
            print_str = f"{args.model} {args.label_str} {args.lmbda} Train epoch {epoch}: [{i * len(p_im)}/{len(train_dataloader.dataset)} ({100. * i / len(train_dataloader):.0f}%)]"
            print_l = print_loss(out_criterion)
            logger.info(print_str + print_l)
            
        if i % args.save_im_interval == 0:

            edge_img = torch.cat([edge_im[:8,:,:,:], edge_im[:8,:,:,:], edge_im[:8,:,:,:]], 1)
            edge_img = torch.nn.functional.interpolate(edge_img, size=[256, 256], mode='nearest', align_corners=None)

            save_images([p_im[:8,:,:,:], base_im[:8,:,:,:],  edge_img, out_net['x_hat'][:8,:,:,:]], fname=f"{save_img_path}/{epoch}-{i}.png", nrow=8)
            

        for kk in out_criterion:
            writer.add_scalar(
                f"Loss/Train/{out_criterion[kk]}",
                out_criterion[kk],
                step)

def train(args):
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataloader = DataLoader_Edge_All(args)

    net = Model()
    net = net.to(args.device)

    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    last_epoch = 0
    if args.restore:  # load from previous checkpoint
        print("Loading", args.restore)
        checkpoint = torch.load( f'{args.ckpt}', map_location=args.device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_one_epoch(
            args,
            net,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            save_img_path
        )
        if args.save and epoch %5 ==0:
            net.update(force=True)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    #"lr_scheduler": lr_scheduler.state_dict(),
                },
                save_path=f"../ckpt/layer3/ckpt/{args.model}/{args.label_str}/{config.args.lmbda}/",
                filename="{:0>4d}.pth.tar".format(epoch),
                is_best=True
            )
     
def test(args):

    
    args.batch_size = 1
    test_dataloader = DataLoader_Edge_All(args)

    dists = DISTS().cuda()
    lpips = LPIPS(
        net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
        version='0.1'  # Currently, v0.1 is supported
    ).cuda()
    
    save_path = f'../results/layer3/img/{args.model}/{args.label_str}/{config.args.lmbda}/'
    save_bin_path = f'../results/layer3/bin/{args.model}/{args.label_str}/{config.args.lmbda}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_bin_path):
        os.makedirs(save_bin_path)


    convert_tensor = transforms.ToTensor()

    net = Model()
    net.cuda()

    checkpoint = torch.load(args.ckpt)
    #for k,v in checkpoint["state_dict"].items():
    #    print(k)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()


    val_list = []
    lpips_list = []
    dists_list = []
    
    bpp_list = []
    save_ims = True
    for i, (p_im, _, img_name, edge_im, base_im) in enumerate(test_dataloader):
        #print(img_name)
        p_im = p_im.cuda()
        edge_im = edge_im.cuda()
        base_im = base_im.cuda()
        

        num_pixels =  p_im.size(0) * p_im.size(2) * p_im.size(3)

        out_net = net.compress(p_im, base_im, edge_im)
        save_bin_file_path = save_bin_path + img_name[0].replace('.jpg', '.bin')
        bpp1 = write_to_bin(out_net, h=p_im.size(2), w=p_im.size(3), save_path=save_bin_file_path, lmbda=args.lmbda)
        bits_bin = os.path.getsize(save_bin_file_path)
        bits_bin = bits_bin * 8
        pixel_bits = bits_bin / num_pixels
        bpp_list.append(pixel_bits)

        strings, original_size, shape, lmbda = read_from_bin(save_bin_file_path)

        out_dec = net.decompress(strings, shape, edge_im, base_im) #lmbda is used to load different models.
        de_img = out_dec["x_hat"]

        
        ndarr = (0.5*(de_img[0]+1)*255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        
        im = Image.fromarray(ndarr)
        gt_path = args.test_imgs_path + img_name[0]
        
        im.save(f"{save_path}/{img_name[0]}")

        if save_ims:
            img1 = cv2.imread(gt_path)
            img2 = cv2.imread(f"{save_path}/{img_name[0]}")
            psnr = cv2.PSNR(img1, img2)
            val_list.append(psnr)

            or_img = convert_tensor(Image.open(gt_path)).cuda()
            de_img = convert_tensor(Image.open(f"{save_path}/{img_name[0]}")).cuda()
            dists_value = dists(or_img.unsqueeze(0), de_img.unsqueeze(0)).cpu().detach().numpy()
            dists_list.append(dists_value)

            lpips_value = lpips(or_img, de_img).cpu().detach().numpy()
            lpips_value = lpips_value[0][0][0][0]
            lpips_list.append(lpips_value)


        save_ims_tmp = False
        if save_ims_tmp:
            edge_img = torch.cat([edge_im[:2,:,:,:], edge_im[:2,:,:,:], edge_im[:2,:,:,:]], 1)
            if not os.path.exists(f'{save_path}/tmp/'):
                os.makedirs(f"{save_path}/tmp/")
            edge_img = torch.nn.functional.interpolate(edge_img, size=[256, 256], mode='nearest', align_corners=None)
            save_images([p_im[:2,:,:,:], base_im[:2,:,:,:], edge_img, out_dec['x_hat'][:2,:,:,:]], fname=f"{save_path}/tmp/{img_name[0]}", nrow=2)

    nn = args.ckpt.split('/')[-1]
    print(f'name: {nn}, bpp:{np.mean(np.array(bpp_list)).round(2)} psnr:{np.mean(np.array(val_list)).round(2)} dists:{np.mean(np.array(dists_list)).round(2)}, lpips:{np.mean(np.array(lpips_list)).round(2)}')

if __name__ == "__main__":
    args = config.args

    if args.mode == 'train':
        train(args)
    elif args.mode == "test":
        test(args)
