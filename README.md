# Segmentation in Medical Image

------

This project aims at providing deep learning models in medical image segmentation.

<img src="image\BraTS19\T1.png" width=20%><img src="image\BraTS19\T1Gd.png" width=20%><img src="image\BraTS19\T2.png" width=20%><img src="image\BraTS19\FLAIR.png" width=20%><img src="image\BraTS19\tumor.png" width=20%>



### Dependencies

------

- pytorch
- torchvision
- tensorboard_logger
- SimpleITK (optional)



### Usage

Prepare the dataset

- ##### Train

```
python train.py
```



### Customize

1. Prepare your own dataset and write the `torch.nn.Dataset` for specific data in folder ***data***
2. Design your network model in folder ***model***
3. Design the evaluation procedure in folder ***evaluate***
4. Write your own configuration file in folder ***config***
5. Modify the `dataset_name` to your dataset in **train.py**
6. Finally, run `python train.py` 



### Experiment

------

- ##### ICH210

|       | Dice Coeff (Std.) | IoU (Std.) | Sensitivity (Std.) | Specificity (Std.) |
| ----- | ---------- | ---- | ----------- | ----------- |
| U-Net | 82.69  (2.17) % | 70.55 (3.13) % | 75.99 (3.05) % | 90.76 (1.72) % |

- ##### BraTS19



### License

------

Only academic study