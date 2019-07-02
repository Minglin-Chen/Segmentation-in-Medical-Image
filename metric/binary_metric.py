
# Dice Similarity Coefficient
def calc_dice(tn, fp, fn, tp):

    numerator = 2 * tp
    denominator = 2 * tp + fn + fp

    if denominator == 0:
        dice = 1.0
    else:
        dice = numerator / denominator

    return dice

# Intersection Of Union
def calc_IoU(tn, fp, fn, tp):

    if fn + tp + fp == 0:
        IoU1 = 1.0
    else:
        IoU1 = tp / (fn + tp + fp)

    if fn + tn + fp == 0:
        IoU0 = 1.0
    else:
        IoU0 = tn / (fn + tn + fp)

    mIoU = (IoU1 + IoU0) / 2.0

    return IoU0, IoU1, mIoU

# Sensitivity
def calc_sensitivity(tn, fp, fn, tp):

    if tp + fn == 0:
        sensitivity = 1.0
    else:
        sensitivity = tp / (tp + fn)
    
    return sensitivity

# Specificity
def calc_specificity(tn, fp, fn, tp):

    if fp + tn == 0:
        specificity = 1.0
    else:
        specificity = tn / (fp + tn)

    return specificity