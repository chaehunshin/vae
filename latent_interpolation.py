from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from model import *
from celeba_dataset import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    dataset = celeba_dataset('../dataset/celeba_dataset/img_align_celeba/', 64)
    dataloader = DataLoader(dataset, 20, True, pin_memory=True)
    model = VAE(3, 500, 128).to(device)

    model.load_state_dict(torch.load('./ckpt/model_271.ckpt'))


    for i, (imgs) in enumerate(dataloader):
        imgs1 = imgs[:10].to(device) #Original Images1
        imgs2 = imgs[10:].to(device) #Original Images2

        mu1, _ = model.encode(imgs1)
        mu2, _ = model.encode(imgs2)

        recon_img_arr = []
        for j in range(11):
            mu = mu1 + (mu2-mu1) * j/10
            recon_imgs = model.decode(mu)
            recon_img_arr.append(recon_imgs)

        show_imgs = [imgs1] + recon_img_arr + [imgs2]
        # show_imgs = make_grid(torch.cat(show_imgs, dim=0), 10)
        # show_imgs = make_grid(torch.cat(show_imgs, dim=3), nrow=12)
        show_imgs = torch.cat(show_imgs, dim=3)
        save_image(show_imgs, './result/inter_D_%05d.jpg'%(i+1), nrow=1)
        if i >19:
            break


    for i in range(20):
        z1 = torch.rand(10, 500).to(device)
        z2 = torch.rand(10, 500).to(device)

        recon_img_arr = []
        for j in range(11):
            z = z1 + (z2 - z1) * j / 10
            recon_img_arr.append(model.decode(z))

        show_imgs = torch.cat(recon_img_arr, dim=3)
        save_image(show_imgs, './result/inter_R_%05d.jpg'%(i), nrow=1)



if __name__=='__main__':
    main()






