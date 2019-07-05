import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from cv2 import cv2

"""
Label modified 
    Origin:
        0 background
        1 NCR(Necrotic) + NET(Non-Enhancing Tumor Core)
        2 ED(Edema)
        4 ET(Enhancing Tumor)
    Target:
        0 background
        1 NCR(Necrotic) + NET(Non-Enhancing Tumor Core)
        2 ED(Edema)
        3 ET(Enhancing Tumor)
The sub-regions considered for evaluation
    1 Enhancing Tumor (ET)                                      [T1Gd]
    2 Tumor Core (TC) = Enhancing Tumor (ET) + NCR & NET        [T1Gd]
    3 Whole Tumor (WT) = Tumor Core (TC) + Edema (ED)           [FLAIR]
"""
class BraTS19_2D(Dataset):

    def __init__(self, dataset_root, categories='HGG', is_train=True):
        """
            dataset_root: direcotory
            categories: 'HGG', 'LGG' or 'Both'
            is_train: train or validation dataset
        """

        with open(os.path.join(dataset_root, '2D', 'trainval_{}.json'.format(categories))) as f:
            trainval_set = json.load(f)
        paths = trainval_set['train' if is_train else 'val']

        self.fullpaths = [ os.path.join(dataset_root, '2D', item) for item in paths ]

    def __len__(self):
        return len(self.fullpaths)

    def __getitem__(self, index):

        # 1. Path
        path = self.fullpaths[index]

        # 2. Load
        npzfile = np.load(path)
        image, seg_annotation = npzfile['MR'], npzfile['lesion']
        
        image = torch.tensor(image, dtype=torch.float32)
        seg_annotation = torch.tensor(seg_annotation, dtype=torch.long)

        # 3. Merge
        image = image
        target = seg_annotation

        return image, target

if __name__=='__main__':

    # 1. Load dataset
    dataset_train = BraTS19_2D(dataset_root='../', is_train=True)
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2)

    # 2. Loop
    for i, (images, targets) in enumerate(dataloader_train):

        print('images shape {} , dtype {}, range {} - {}'.format(images.shape, images.dtype, images.min(), images.max()))
        print('targets shape {} , dtype {}, range {} - {}'.format(targets.shape, targets.dtype, targets.min(), targets.max()))

        T1s_tensor = make_grid(images[:,0:1], nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        T1s_np = T1s_tensor.numpy().transpose((1,2,0))

        T1Gds_tensor = make_grid(images[:,1:2], nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        T1Gds_np = T1Gds_tensor.numpy().transpose((1,2,0))

        T2s_tensor = make_grid(images[:,2:3], nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        T2s_np = T2s_tensor.numpy().transpose((1,2,0))

        FLAIRs_tensor = make_grid(images[:,3:4], nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        FLAIRs_np = FLAIRs_tensor.numpy().transpose((1,2,0))

        targets = targets.float() / 3.0
        targets_tensor = make_grid(targets.unsqueeze(dim=1), nrow=4, padding=12, normalize=True, pad_value=1.0)
        targets_np = targets_tensor.numpy().transpose((1,2,0)).astype(np.float)

        cv2.imshow('T1', T1s_np)
        cv2.imshow('T1Gd', T1Gds_np)
        cv2.imshow('T2', T2s_np)
        cv2.imshow('FLAIR', FLAIRs_np)
        cv2.imshow('targets', targets_np)
        cv2.waitKey()

        if i > 3:
            break