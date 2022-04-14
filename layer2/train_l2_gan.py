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

from data.dataloader import DataLoader_Edge_All 
from loss.gan_loss import gan_loss

from torchvision import transforms
from DISTS_pytorch import DISTS
from lpips_pytorch import LPIPS


from model.DNet import Discriminator

from Common.util import save_images
from PIL import Image

Tensor = torch.cuda.FloatTensor
#torch.autograd.set_detect_anomaly(True)

config.args.model = "L2_MS_Edge_Semantic"
pkg = importlib.import_module('model.'+config.args.model)
Model =  getattr(pkg, config.args.model)


logger = logging.getLogger(config.args.model)
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
stdhandler = logging.StreamHandler()
stdhandler.setFormatter(formatter)
logger.addHandler(stdhandler)

if not os.path.exists('../ckpt/layer2/log/'):
    os.makedirs('../ckpt/layer2/log/')

filehandler = logging.FileHandler(f'../ckpt/layer2/log/{config.args.model}_{config.args.label_str}.log')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

writer = SummaryWriter(f'../ckpt/layer2/log/{config.args.model}/{config.args.label_str}/')
save_img_path = f"../ckpt/layer2/figures/{config.args.model}/{config.args.label_str}/"

step = 0

if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
        
def configure_optimizers(net, dnet, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    discriminator_parameters = dnet.parameters()
    dnet_optimizer = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)

    return optimizer, dnet_optimizer

def train_one_epoch(args, model, DNet, train_dataloader, optimizer, dnet_optimizer, epoch, save_img_path):
    global step
    model.train()
    device = next(model.parameters()).device
    
    for i, (p_im, _, caps, _, edge_im, style_im) in enumerate(train_dataloader):

        p_im = p_im.to(device)
        edge_im = edge_im.to(device)
        style_im = style_im.to(device)

        optimizer.zero_grad()

        out_net = model(edge_im, style_im)
        out_criterion = model.loss(out_net, p_im)

        D_in = torch.cat([p_im, out_net["x_hat"]], dim=0)
        D_out, D_out_logits = DNet(D_in)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        #G_loss = gan_loss(D_real, D_gen, D_real_logits, D_gen_logits, gan_loss_type='least_squares', mode='generator_loss')
        G_loss = gan_loss(D_real, D_gen, D_real_logits, D_gen_logits, gan_loss_type='non_saturating', mode='generator_loss')
        out_criterion["G_loss"] = G_loss 
        out_criterion["loss"] = out_criterion["loss"] + out_criterion["G_loss"] 
        
        out_criterion["loss"].backward(retain_graph=True)
        optimizer.step()

        dnet_optimizer.zero_grad()
        D_in = torch.cat([p_im, out_net["x_hat"].detach()], dim=0)
        D_out, D_out_logits = DNet(D_in)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)
        
        #D_loss = gan_loss(D_real, D_gen, D_real_logits, D_gen_logits, gan_loss_type='least_squares', mode='discriminator_loss')
        D_loss = gan_loss(D_real, D_gen, D_real_logits, D_gen_logits, gan_loss_type='non_saturating', mode='discriminator_loss')

        
        out_criterion["D_loss"] = D_loss
        out_criterion["D_loss"].backward()
        dnet_optimizer.step()


        step = step + 1
        if i % args.print_interval == 0:
            print_str = f"{args.model} {args.label_str} Train epoch {epoch}: [{i * len(p_im)}/{len(train_dataloader.dataset)} ({100. * i / len(train_dataloader):.0f}%)]"
            print_l = print_loss(out_criterion)
            logger.info(print_str + print_l)
            
        if i % args.save_im_interval == 0:

            edge_img = torch.cat([edge_im[:8,:,:,:], edge_im[:8,:,:,:], edge_im[:8,:,:,:]], 1)
            edge_img = torch.nn.functional.interpolate(edge_img, size=[256, 256], mode='nearest', align_corners=None)
            save_images([p_im[:8,:,:,:], style_im[:8,:,:,:], edge_img, out_net['x_hat'][:8,:,:,:]], fname=f"{save_img_path}/{epoch}-{i}.png")

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

    dnet = Discriminator(image_dims=[3, args.image_size, args.image_size])
    dnet = dnet.to(args.device)
    
    optimizer, dnet_optimizer = configure_optimizers(net, dnet, args)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    last_epoch = 0
    if args.restore:  # load from previous checkpoint
        print("Loading", args.restore)
        
        checkpoint = torch.load( f'{args.ckpt}', map_location=args.device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        dnet_optimizer.load_state_dict(checkpoint["dnet_optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        #optimizer.param_groups[0]['lr'] = 0.0001
        train_one_epoch(
            args,
            net,
            dnet,
            train_dataloader,
            optimizer,
            dnet_optimizer,
            epoch,
            save_img_path
        )
        
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dnet_optimizer": dnet_optimizer.state_dict(),
                    #"lr_scheduler": lr_scheduler.state_dict(),
                },
                save_path=f"../ckpt/layer2/model/{args.model}/{args.label_str}/",
                filename="{:0>4d}.pth.tar".format(epoch),
                is_best=True
            )
        #lr_scheduler.step(loss)
        
  
def test(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    test_dataloader = DataLoader_Edge_All(args)

    dists = DISTS().cuda()
    lpips = LPIPS(
        net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
        version='0.1'  # Currently, v0.1 is supported
    ).cuda()
    
    save_path = f'../results/layer2/img/{args.model}/{args.label_str}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net = Model()
    net.cuda()

    convert_tensor = transforms.ToTensor()

    checkpoint = torch.load(args.ckpt)

    model_dict = net.state_dict()
    pretrained_dict = checkpoint['state_dict']
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    for k in new_dict:
        print(k, '--')
    model_dict.update(new_dict)
    net.load_state_dict(model_dict)
    #net.load_state_dict(checkpoint['state_dict'])
    for parameter in net.parameters():
        parameter.requires_grad = False

    net.eval()
    val_list = []
    lpips_list = []
    dists_list = []
    save_ims = True
    for i, (p_im, _, caps, img_name, edge_im, style_im) in enumerate(test_dataloader):

        p_im = p_im.cuda()
        edge_im = edge_im.cuda()
        style_im = style_im.cuda()

        out = net(edge_im, style_im)
        de_img = out["x_hat"]
        
        for gt_im, img, name in zip(p_im, de_img, img_name):
            ndarr = (0.5*(img+1)*255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            gt_path = args.test_imgs_path + name

            im.save(f"{save_path}/{name}")
            print(f"{save_path}/{name}/")
            if save_ims:
                or_img = convert_tensor(Image.open(gt_path)).cuda()
                de_img = convert_tensor(Image.open(f"{save_path}/{name}")).cuda()
                lpips_value = lpips(or_img, de_img)
                dists_value = dists(or_img.unsqueeze(0), de_img.unsqueeze(0)).cpu().detach().numpy()
                dists_list.append(dists_value)

                lpips_value = lpips(or_img, de_img).cpu().detach().numpy()
                lpips_value = lpips_value[0][0][0][0]
                lpips_list.append(lpips_value)

                print(f"{dists_value:0<2.2f}---{lpips_value:0<2.2f}-{name}")

        save_ims_flag = False
        if save_ims_flag:
            edge_img = torch.cat([edge_im[:2,:,:,:], edge_im[:2,:,:,:], edge_im[:2,:,:,:]], 1)
            if not os.path.exists(f'{save_path}/tmp/'):
                os.makedirs(f"{save_path}/tmp/")
                
            edge_img = torch.nn.functional.interpolate(edge_img, size=[256, 256], mode='nearest', align_corners=None)
            save_images([p_im[:2,:,:,:], style_im[:2,:,:,:], edge_img, out['x_hat'][:2,:,:,:]], fname=f"{save_path}/tmp/{name}", nrow=2)

    nn = args.ckpt.split('/')[-1]
    print(f'name: {nn}, dists:{np.mean(np.array(dists_list)).round(2)}, lpips:{np.mean(np.array(lpips_list)).round(2)}')

if __name__ == "__main__":
    args = config.args
    
    if args.mode == 'train':
        train(args)
    elif args.mode == "test":
        test(args)
