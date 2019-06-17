import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import argparse
import os
import numpy as np

from model import VAE
from celeba_dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    dataset = celeba_dataset(args.data_path, args.img_size)

    logger = SummaryWriter(args.logdir)

    dataloader = DataLoader(dataset, args.batch_size, True, pin_memory=True)
    model = VAE(3, 500, 128).to(device)

    if os.path.exists(os.path.join(args.ckpt_path, 'model_last.ckpt')):
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'model_last.ckpt')))
        print("model is loaded from saved checkpoint")
    model.train()

    optimizer = optim.Adam(model.parameters(), args.lr)

    global_step = 0
    for epoch in range(31, args.total_epoch):
        for i, imgs in enumerate(dataloader):
            global_step += 1

            imgs = imgs.to(device)

            recon_imgs, mu, logvar = model(imgs)

            recon_loss = torch.nn.BCELoss(reduction='sum')(recon_imgs.reshape(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1))
            # recon_loss = F.binary_cross_entropy(recon_imgs.view(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1))
            # recon_loss = F.mse_loss(recon_imgs.view(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1))
            # kld_loss = torch.mean(-0.5*torch.sum(1 + logvar - mu **2 - torch.exp(logvar), dim=1))/(64*64)
            kld_loss = -0.5 * torch.sum((1 + logvar - mu**2 - torch.exp(logvar)))
            total_loss = recon_loss + kld_loss

            # assert not torch.isnan(total_loss).any() print("What is the problem?")

            # if torch.isnan(total_loss).any():
            #     print('stop!')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (global_step + 1) % args.log_freq == 0 :
                orig_imgs = make_grid(imgs[:16], 4, normalize=True)
                recon_imgs = make_grid(recon_imgs[:16], 4, normalize=True)

                logger.add_scalars('loss', {'recon_loss': recon_loss.data.item(),
                                            'kld_loss': kld_loss.data.item(),
                                            'total_loss': total_loss.data.item()}, global_step)
                logger.add_image('images', orig_imgs, global_step)
                logger.add_image('recon_imgs', recon_imgs, global_step)

            print("Epoch: %d/%d, Step: %d/%d, Recon: %.4f, KLD: %.4f"%(epoch+1, args.total_epoch, i+1, len(dataloader), recon_loss.data.item(), kld_loss.data.item()))

        if (epoch) % args.save_freq == 0 :
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'model_%d.ckpt'%(epoch+1)))
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'model_lastk.cpkt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the vae with celeba')

    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--total_epoch', default=600)

    parser.add_argument('--data_path', default='../dataset/celeba_dataset/img_align_celeba/')
    parser.add_argument('--ckpt_path', default='./ckpt')
    parser.add_argument('--logdir', default='./logs/')
    parser.add_argument('--save_freq', default=10)
    parser.add_argument('--log_freq', default=50)

    args = parser.parse_args()
    main(args)
