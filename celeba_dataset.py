from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

import os
import random

class celeba_dataset(Dataset):
    def __init__(self, root_dir, resize_dim, shuffle=True, transform=None):
        self._root_dir = root_dir
        self._resize_dim = resize_dim
        if transform is not None:
            self._transform = transform
        else:
            self._transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize((resize_dim, resize_dim)),
                T.ToTensor(),
                # T.Normalize(mean=[0.5, 0.5, 0.5],
                #             std=[0.5, 0.5, 0.5])
            ])
        self._im_list = os.listdir(root_dir)

        if shuffle:
            random.shuffle(self._im_list)

    def __len__(self):
        return len(self._im_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self._root_dir, self._im_list[idx]))
        return self._transform(image)



