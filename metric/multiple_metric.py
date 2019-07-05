
# Dice Similarity Coefficient
def calc_dice_from_matrix(count_matrix):

    dice_coeff_dict = {}
    total_count = 0.0

    num_class = count_matrix.shape[0]
    for i_cls in range(num_class):

        numerator = 2 * count_matrix[i_cls, i_cls]
        denominator = count_matrix[i_cls, :].sum() + count_matrix[:,i_cls].sum()

        if denominator == 0.0:
            dice = 1.0
        else:
            dice = numerator / denominator

        dice_coeff_dict['Dice_Coeff_{}'.format(i_cls)] = dice
        total_count += dice

    dice_coeff_dict['Mean_Dice_Coeff'] = total_count / num_class

    return dice_coeff_dict

# Intersection Of Union
def calc_IoU_from_matrix(count_matrix):

    IoU_dict = {}
    total_count = 0.0

    num_class = count_matrix.shape[0]
    for i_cls in range(num_class):

        numerator = count_matrix[i_cls, i_cls]
        denominator = count_matrix[i_cls, :].sum() + count_matrix[:,i_cls].sum() - count_matrix[i_cls,i_cls]

        if denominator == 0.0:
            IoU = 1.0
        else:
            IoU = numerator / denominator

        IoU_dict['IoU_{}'.format(i_cls)] = IoU
        total_count += IoU

    IoU_dict['Mean_IoU'] = total_count / num_class

    return IoU_dict

# Sensitivity
def calc_sensitivity_from_matrix(count_matrix):

    sensitivity_dict = {}
    total_count = 0.0

    num_class = count_matrix.shape[0]
    for i_cls in range(num_class):

        numerator = count_matrix[i_cls,i_cls]
        denominator = count_matrix[i_cls, :].sum()

        if denominator == 0.0:
            sensitivity = 1.0
        else:
            sensitivity = numerator / denominator

        sensitivity_dict['Sensitivity_{}'.format(i_cls)] = sensitivity
        total_count += sensitivity

    sensitivity_dict['Mean_Sensitivity'] = total_count / num_class

    return sensitivity_dict

# Specificity
def calc_specificity_from_matrix(count_matrix):

    specificity_dict = {}
    total_count = 0.0

    num_class = count_matrix.shape[0]
    for i_cls in range(num_class):

        numerator = count_matrix[i_cls, i_cls]
        denominator = count_matrix[:, i_cls].sum()

        if denominator == 0.0:
            specificity = 1.0
        else:
            specificity = numerator / denominator

        specificity_dict['Specificity_{}'.format(i_cls)] = specificity
        total_count += specificity

    specificity_dict['Mean_Specificity'] = total_count / num_class

    return specificity_dict