import torchvision.transforms as T

from .ICH210_2D import ICH210_2D

def dataset_provider(name, dataset_root, is_train=True, fold_idx=0):

    dataset = None

    if name == 'ICH210_2D':

        dataset = ICH210_2D(
                dataset_root=dataset_root, 
                only_hemorrhage=True, 
                is_train=is_train, 
                fold_idx=fold_idx)
    
    return dataset