# import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# from PIL import Image
import os
from scipy.misc import imsave

from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    if not os.path.exists('./result'):
        os.makedirs('./result')

    model = VAE(3, 500, 128).to(device)

    model.load_state_dict(torch.load('ckpt/model_271.ckpt'))

    for i in range(20):
        z = torch.rand(32, 500).to(device)
        generated_img = model.decode(z)

        generated_img = make_grid(generated_img, normalize=True)
        # generated_img = (generated_img.cpu().detach().numpy().transpose(1, 2, 0)*255).astype('uint8')
        # imsave('./result/results_%05d.jpg'%(i+1), generated_img)
        save_image(generated_img, './result/result_%05d.jpg'%(i+1))



if __name__=='__main__':
    main()
