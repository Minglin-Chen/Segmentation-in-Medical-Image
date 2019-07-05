"""
    This file used to configurate the train procedure
"""

import torch.optim as optim

def get_config_from_dataset(dataset_name):

    if dataset_name == 'ICH210_2D':
        _config = {
            'dataset_name': dataset_name,
            'dataset_root': '../../data/DatasetICH/ICH210',
            'batch_size': 40, 
            'num_workers': 8,
            'model': 'UNet',
            'image_channels': 1,
            'num_class': 2, 
            'criterion': 'ce',
            'optimizer': optim.Adam,
            'lr_scheduler': False,
            'lr': 1e-3,
            'eval_op_name': 'eval_op_ICH210',
            'num_epoch': 100,
            'device_ids': [0,1],
            'log_image': False,
            'num_folds': 5 }

    elif dataset_name == 'BraTS19_2D':
        _config = {
            'dataset_name': dataset_name,
            'dataset_root': '../../data/BraTS19',
            'batch_size': 40, 
            'num_workers': 8,
            'model': 'UNet',
            'image_channels': 4,
            'num_class': 4, 
            'criterion': 'hybrid',
            'optimizer': optim.Adam,
            'lr_scheduler': False,
            'lr': 1e-3,
            'eval_op_name': 'eval_op_BraTS19',
            'num_epoch': 100,
            'device_ids': [0],
            'log_image': False,
            'num_folds': 1 }
        
    return _config
