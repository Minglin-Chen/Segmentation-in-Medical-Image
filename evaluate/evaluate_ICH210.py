import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid

from metric import multiple_metric

def eval_op_ICH210(net, dataloader, device_id, criterion, num_class, epoch, logger, log_image):

    net.eval()

    loss = 0.0
    count_matrix = np.zeros((num_class, num_class))

    # go through the dataset
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader)):

            # 0. get the inputs
            images = images.cuda(device_id)
            targets = targets.cuda(device_id)

            # 1. forward
            out = net(images)

            # 2. get probability and prediction
            prediction = torch.argmax(out, dim=1)

            # 3. statisticZ
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

    # Loss
    loss /= len(dataloader)
    logger.log_value('loss', loss, step=epoch)

    # Dice Coefficient
    dice_coeff_dict = multiple_metric.calc_dice_from_matrix(count_matrix)
    for k, v in dice_coeff_dict.items():
        logger.log_value(k, v, step=epoch)

    # IoU
    IoU_dict = multiple_metric.calc_IoU_from_matrix(count_matrix)
    for k, v in IoU_dict.items():
        logger.log_value(k, v, step=epoch)

    # Sensitivity
    sensitivity_dict = multiple_metric.calc_sensitivity_from_matrix(count_matrix)
    for k, v in sensitivity_dict.items():
        logger.log_value(k, v, step=epoch)

    # Specificity
    specificity_dict = multiple_metric.calc_specificity_from_matrix(count_matrix)
    for k, v in specificity_dict.items():
        logger.log_value(k, v, step=epoch)

    # Print
    dice = dice_coeff_dict['Dice_Coeff_1']
    IoU = IoU_dict['IoU_1']
    sensitivity = sensitivity_dict['Sensitivity_1']
    specificity = specificity_dict['Specificity_1']

    print('Evaluation Loss {:.4f} Dice Coefficient {:.4f}'.format(loss, dice))

    return dice, IoU, sensitivity, specificity