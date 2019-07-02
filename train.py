import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Data
from data import dataset_provider
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, WeightedRandomSampler
# Model
from model.UNet import UNet
# Loss
from loss import criterion_provider
# Optimization
import torch.optim as optim
# Metric
from metric import binary_metric
# Log
from tensorboard_logger import Logger
from tqdm import tqdm
from torchvision.utils import make_grid
import json

# Configuration
config = {
    'dataset_name': 'ICH210_2D',
    'dataset_root': '../../data/DatasetICH/ICH210',
    'batch_size': 40, 
    'num_workers': 8,
    'criterion': 'hybrid',
    'lr': 1e-3,
    'num_epoch': 100,
    'device_ids': [0,1],
    'log_image': False
}

if not os.path.exists('weights'):
    os.mkdir('weights')

def train_op(net, dataloader, criterion, optimizer, scheduler, epoch, logger):

    net.train()

    running_loss = 0.0
    start_t = time.time()
    for i, (images, targets) in enumerate(dataloader):

        # 0. get the inputs
        images = images.cuda(config['device_ids'][0])
        targets = targets.cuda(config['device_ids'][0])

        # 1. calculate loss
        probs = net(images)
        loss = criterion(probs, targets)

        # 2. update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 3. logger
        running_loss += loss.item()
        n = 1
        if i % n == n-1:
            print('[Epoch {:0>3} Step {:0>3}/{:0>3}] Loss {:.4f} Time {:.2f} s'.format(
                epoch+1, i+1, len(dataloader), running_loss/n, time.time()-start_t))
            logger.log_value('loss', running_loss/n, step=i+epoch*len(dataloader))
            # reinitialization
            running_loss = 0.0
            start_t = time.time()

def eval_op(net, dataloader, criterion, epoch, logger, log_image):

    net.eval()

    loss = 0.0
    # [[ TN, FP ], [ FN, TP ]]
    count_matrix = np.zeros((2, 2)) 

    # go through the dataset
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader)):

            # 0. get the inputs
            images = images.cuda(config['device_ids'][0])
            targets = targets.cuda(config['device_ids'][0])

            # 1. forward
            out = net(images)

            # 2. get probability and prediction
            probability = F.softmax(out, dim=1)[:,1:2]
            prediction = torch.argmax(out, dim=1)

            # 3. statistic
            prediction_flat = prediction.view(-1).cpu().numpy()
            target_flat = targets.view(-1).cpu().numpy()
            for j in range(target_flat.shape[0]):
                count_matrix[int(target_flat[j]), int(prediction_flat[j])] += 1

            loss += criterion(out, targets).item()

            # 4. qualitative comparison
            if log_image:
                images_tensor = make_grid(images, nrow=5, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
                images_np = images_tensor.cpu().numpy().transpose((1,2,0))
                logger.log_images('Eval Images {}'.format(i), [images_np], step=0)

                targets_tensor = make_grid(targets.unsqueeze(dim=1), nrow=5, padding=12, normalize=False, pad_value=1.0)
                targets_np = targets_tensor.cpu().numpy().transpose((1,2,0)).astype(np.float)
                logger.log_images('Eval Targets {}'.format(i), [targets_np], step=0)

                prediction_tensor = make_grid(prediction.unsqueeze(dim=1), nrow=5, padding=12, normalize=False, pad_value=1.0)
                prediction_np = prediction_tensor.cpu().numpy().transpose((1,2,0)).astype(np.float)
                logger.log_images('Eval Prediction {}'.format(i), [prediction_np], step=0)

                probability_tensor = make_grid(probability, nrow=5, padding=12, normalize=False, pad_value=1.0)
                probability_np = probability_tensor.cpu().numpy().transpose((1,2,0))
                logger.log_images('Eval Probability {}'.format(i), [probability_np], step=0)

    # Performance
    tn, fp, fn, tp = count_matrix[0,0], count_matrix[0,1], count_matrix[1,0], count_matrix[1,1]
    # Dice Coefficient
    dice_coefficient = binary_metric.calc_dice(tn, fp, fn, tp)
    logger.log_value('Dice Coefficient', dice_coefficient, step=epoch)
    # IoU
    IoU0, IoU1, mIoU = binary_metric.calc_IoU(tn, fp, fn, tp)
    logger.log_value('IoU1', IoU1, step=epoch)
    logger.log_value('IoU0', IoU0, step=epoch)
    logger.log_value('mIoU', mIoU, step=epoch)
    # Sensitivity
    sensitivity = binary_metric.calc_sensitivity(tn, fp, fn, tp)
    logger.log_value('Sensitivity', sensitivity, step=epoch)
    # Specificity
    specificity = binary_metric.calc_specificity(tn, fp, fn, tp)
    logger.log_value('Specificity', specificity, step=epoch)
    # Loss
    loss /= len(dataloader)
    logger.log_value('loss', loss, step=epoch)

    # Logger
    print('Evaluation Loss {:.4f} Dice Coefficient {:.4f}'.format(loss, dice_coefficient))

    return dice_coefficient, IoU1, sensitivity, specificity

def update_dataloader(net, dataset, epoch, logger):

    net.eval()
    dataloader = DataLoader(dataset, config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    weights = []
    # go through the dataset
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader)):

            # 0. get the inputs
            images = images.cuda(config['device_ids'][0])
            targets = targets.cuda(config['device_ids'][0])

            # 1. forward
            out = net(images)

            # 2. get prediction
            prediction = torch.argmax(out, dim=1)
            N = prediction.shape[0]

            # 3. statistic
            prediction_flat = prediction.view(N, -1).cpu().numpy()
            target_flat = targets.view(N, -1).cpu().numpy()
            for n in range(N):
                count_matrix = np.zeros((2, 2))
                for j in range(target_flat.shape[1]):
                    count_matrix[int(target_flat[n, j]), int(prediction_flat[n, j])] += 1
                tn, fp, fn, tp = count_matrix[0,0], count_matrix[0,1], count_matrix[1,0], count_matrix[1,1]
                dice_coefficient = binary_metric.calc_dice(tn, fp, fn, tp)
                
                weights.append( 1-dice_coefficient )

    # Update
    sampler = WeightedRandomSampler(weights, len(dataset))
    dataloader = DataLoader(dataset, config['batch_size'], sampler=sampler, num_workers=config['num_workers'])

    return dataloader

def trainer(index=0):

    # 1. Load dataset
    dataset_train = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=True, fold_idx=index)
    dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    dataloader_eval = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=False, fold_idx=index)
    dataloader_eval = DataLoader(dataloader_eval, 10, shuffle=False, num_workers=config['num_workers'])

    # 2. Build model
    net = UNet().cuda(config['device_ids'][0])
    net = nn.DataParallel(net, device_ids=config['device_ids'])

    # 3. Criterion
    criterion = criterion_provider(config['criterion'], weight=torch.tensor([0.0, 1.0])).cuda(config['device_ids'][0])

    # 4. Optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    step_size = len(dataloader_train) * config['num_epoch'] // (4 * 21) + 1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size , gamma=0.9)

    # 5. Tensorboard logger
    logger_train = Logger('logs/fold_{}/train'.format(index))
    logger_eval = Logger('logs/fold_{}/eval'.format(index))

    # 6. Train loop
    dice_max, IoU1_max, sensitivity_max, specificity_max = -1.0, -1.0, -1.0, -1.0
    for epoch in range(config['num_epoch']):

        # train
        print('---------------------- Train ----------------------')
        train_op(net, dataloader_train, criterion, optimizer, scheduler, epoch, logger_train)

        # resample
        # if epoch % 20 == 19:
        #     print('---------------------- Update Sampler ----------------------')
        #     dataloader_train = update_dataloader(net, dataset_train, epoch, logger_eval)

        # evaluation
        if epoch % 10 == 9:
            print('---------------------- Evaluation ----------------------')
            dice_coefficient, IoU1, sensitivity, specificity = \
                eval_op(net, dataloader_eval, criterion, epoch, logger_eval, config['log_image'])

        # Update maxinum and Save
        if epoch % 10 == 9:
            torch.save(net.state_dict(), 'weights/{}.newest.{}.pkl'.format(net.module.__class__.__name__, index))
            if dice_max <= dice_coefficient:
                dice_max = dice_coefficient
                torch.save(net.state_dict(), 'weights/{}.best.{}.pkl'.format(net.module.__class__.__name__, index))
            if IoU1_max <= IoU1: IoU1_max = IoU1
            if sensitivity_max <= sensitivity: sensitivity_max = sensitivity
            if specificity_max <= specificity: specificity_max = specificity

    return dice_max, IoU1_max, sensitivity_max, specificity_max, dice_coefficient, IoU1, sensitivity, specificity

if __name__=='__main__':

    # Setting seed
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    np.random.seed(666)
    random.seed(666)
    torch.backends.cudnn.deterministic = True

    # Record
    dice_max_list, IoU1_max_list, sensitivity_max_list, specificity_max_list = [], [], [], []
    dice_list, IoU1_list, sensitivity_list, specificity_list = [], [], [], []

    # K-Fold Cross Validation
    for i in range(5):
        dice_max, IoU1_max, sensitivity_max, specificity_max, dice_coefficient, IoU1, sensitivity, specificity = \
            trainer(i)
        
        # record
        dice_max_list.append(dice_max)
        IoU1_max_list.append(IoU1_max)
        sensitivity_max_list.append(sensitivity_max)
        specificity_max_list.append(specificity_max)

        dice_list.append(dice_coefficient)
        IoU1_list.append(IoU1)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    # Output
    content = {
        'dice_max': dice_max_list,
        'IoU1_max': IoU1_max_list,
        'sensitivity_max': sensitivity_max_list,
        'specificity_max': specificity_max_list,
        'dice': dice_list,
        'IoU1': IoU1_list,
        'sensitivity': sensitivity_list,
        'specificity': specificity_list }

    with open('cv_perf_result.json', 'w') as f:
        json.dump(content, f)
    