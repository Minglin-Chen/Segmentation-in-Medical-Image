import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
# Configuration
from config import get_config_from_dataset
# Data
from data import dataset_provider
from torch.utils.data import DataLoader
# Model
from model import model_provider
# Loss
from loss import criterion_provider
# Optimization
import torch.optim as optim
# Evaluation
from evaluate import eval_op_provider
# Log
import time
from tensorboard_logger import Logger
import json

# Configuration
config = get_config_from_dataset(dataset_name='BraTS19_2D')

time_id = time.strftime('%Y%m%d_%H%M%S')
if not os.path.exists('weights/{}'.format(time_id)):
    os.makedirs('weights/{}'.format(time_id))

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
        if scheduler is not None:
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

def trainer(index=0):

    # 1. Load dataset
    dataset_train = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=True, fold_idx=index)
    dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    dataloader_eval = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=False, fold_idx=index)
    dataloader_eval = DataLoader(dataloader_eval, 10, shuffle=False, num_workers=config['num_workers'])

    # 2. Build model
    net = model_provider(config['model'], img_chn=config['image_channels'], n_cls=config['num_class']).cuda(config['device_ids'][0])
    net = nn.DataParallel(net, device_ids=config['device_ids'])

    # 3. Criterion
    criterion = criterion_provider(config['criterion']).cuda(config['device_ids'][0])

    # 4. Optimizer
    optimizer = config['optimizer'](net.parameters(), lr=config['lr'])
    scheduler = None
    if config['lr_scheduler']:
        step_size = len(dataloader_train) * config['num_epoch'] // (4 * 21) + 1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size , gamma=0.9)

    # 5. Evaluation
    eval_op = eval_op_provider(config['eval_op_name'])

    # 6. Tensorboard logger
    logger_train = Logger('logs/{}/{}/fold_{}/train'.format(config['model'], time_id, index))
    logger_eval = Logger('logs/{}/{}/fold_{}/eval'.format(config['model'], time_id, index))

    # 7. Train loop
    dice_max = -1
    for epoch in range(config['num_epoch']):

        # train
        print('---------------------- Train ----------------------')
        train_op(net, dataloader_train, criterion, optimizer, scheduler, epoch, logger_train)

        # evaluation
        if epoch % 10 == 9:
            print('---------------------- Evaluation ----------------------')
            dice, IoU, sensitivity, specificity = \
                eval_op(net, dataloader_eval, config['device_ids'][0], criterion, config['num_class'], epoch, logger_eval, config['log_image'])
            
            # save weights
            torch.save(net.state_dict(), 'weights/{}/{}.newest.{}.pkl'.format(time_id, config['model'], index))
            if dice >= dice_max:
                dice_max = dice
                torch.save(net.state_dict(), 'weights/{}/{}.best.{}.pkl'.format(time_id, config['model'], index))

    return dice, IoU, sensitivity, specificity

if __name__=='__main__':

    # Setting seed
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    np.random.seed(666)
    random.seed(666)
    torch.backends.cudnn.deterministic = True

    # Record
    dice_list, IoU_list, sensitivity_list, specificity_list = [], [], [], []

    # K-Fold Cross Validation
    for i in range(config['num_folds']):
        dice, IoU, sensitivity, specificity = trainer(i)
        
        # record
        dice_list.append(dice)
        IoU_list.append(IoU)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    # Output
    content = {
        'dice': dice_list,
        'IoU1': IoU_list,
        'sensitivity': sensitivity_list,
        'specificity': specificity_list }

    with open('logs/{}/{}/perf_result.json'.format(config['model'], time_id), 'w') as f:
        json.dump(content, f)
    
