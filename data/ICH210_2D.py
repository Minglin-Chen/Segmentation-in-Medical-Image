import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from cv2 import cv2

"""
0: background
1: hemorrhage
2: edema
"""
class ICH210_2D(Dataset):

    def __init__(self, dataset_root, only_hemorrhage=True, is_train=True, fold_idx=0):

        self.only_hemorrhage = only_hemorrhage

        with open(os.path.join(dataset_root, '2D', 'trainval_5folds.json')) as f:
            folds = json.load(f)
        paths = folds[str(fold_idx)]['train' if is_train else 'val']

        self.scan_fullpaths = [ os.path.join(dataset_root, '2D', 'image', item) for item in paths ]
        self.edema_fullpaths = [ os.path.join(dataset_root, '2D', 'edema', item) for item in paths ]
        self.hemorrhage_fullpaths = [ os.path.join(dataset_root, '2D', 'hemorrhage', item) for item in paths ]

    def __len__(self):
        return len(self.scan_fullpaths)

    def __getitem__(self, index):

        # 1. Path
        scan_path = self.scan_fullpaths[index]
        edema_path = self.edema_fullpaths[index]
        hemorrhage_path = self.hemorrhage_fullpaths[index]

        # 2. Load
        scan = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
        scan = scan[np.newaxis,:]
        scan = scan.astype(np.float32)
        scan = (scan - 128) / 128.0
        scan = torch.tensor(scan, dtype=torch.float32)

        edema = cv2.imread(edema_path, cv2.IMREAD_GRAYSCALE)
        edema[edema!=0] = 1
        edema = torch.tensor(edema, dtype=torch.long)

        hemorrhage = cv2.imread(hemorrhage_path, cv2.IMREAD_GRAYSCALE)
        hemorrhage[hemorrhage!=0] = 1
        hemorrhage = torch.tensor(hemorrhage, dtype=torch.long)

        # 3. Merge
        image = scan
        target = hemorrhage
        if not self.only_hemorrhage:
            target[edema!=0] = 2

        return image, target

if __name__=='__main__':

    # 1. Load dataset
    dataset_train = ICH210_2D(dataset_root='../../../data/DatasetICH/ICH210', only_hemorrhage=True, is_train=True, fold_idx=1)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2)

    # 2. Loop
    for i, (images, targets) in enumerate(dataloader_train):

        print('images shape {} , dtype {}, range {} - {}'.format(images.shape, images.dtype, images.min(), images.max()))
        print('targets shape {} , dtype {}, range {} - {}'.format(targets.shape, targets.dtype, targets.min(), targets.max()))

        images_tensor = make_grid(images, nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        images_np = images_tensor.numpy().transpose((1,2,0))

        targets_tensor = make_grid(targets.unsqueeze(dim=1), nrow=4, padding=12, normalize=False, pad_value=1.0)
        targets_np = targets_tensor.numpy().transpose((1,2,0)).astype(np.float)

        cv2.imshow('images', images_np)
        cv2.imshow('targets', targets_np)
        cv2.waitKey()

        if i > 3:
            break