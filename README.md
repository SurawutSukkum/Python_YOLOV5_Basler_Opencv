# Python_YOLOV5_Basler_Opencv
## Install
Support python version 3.7.9 
```
pip install opencv-contrib-python
pip install pandas 
pip install labelImg
pip install PyYAML
```
## Google drive
Upload this folder to google drive: `yolo_training` 

`https://github.com/SurawutSukkum/Python_YOLOV5_Basler_Opencv/tree/main/yolo_training`

## Google Colab
Connect to drive.
```
from google.colab import drive
drive.mount('/content/drive')
```

## Google Colab
Install requirements
```
import os
os.chdir('/content/drive/MyDrive/yolo_training')
os.chdir('yolov5')
!pip install -r requirements.txt
```

## Google Colab
Training YOLO
```
!python train.py --data my_obj.yaml --cfg yolov5l.yaml --batch-size 24 --name Model --epochs 3000
!python export.py --weights runs/train/Model/weights/best.pt --include torchscript onnx
print('Training Done')
```
## Google Colab
Training
```
YOLOv5l summary: 468 layers, 46159834 parameters, 46159834 gradients, 108.3 GFLOPs

Transferred 57/613 items from yolov5s.pt
AMP: checks passed ✅
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 101 weight (no decay), 104 weight, 104 bias
albumentations: Blur(always_apply=False, p=0.01, blur_limit=(3, 7)), MedianBlur(always_apply=False, p=0.01, blur_limit=(3, 7)), ToGray(always_apply=False, p=0.01), CLAHE(always_apply=False, p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/drive/MyDrive/yolo_training/yolov5/my_data_images_for_train/train.cache' images and labels... 82 found, 0 missing, 0 empty, 0 corrupt: 100% 82/82 [00:00<?, ?it/s]
val: Scanning '/content/drive/MyDrive/yolo_training/yolov5/my_data_images_for_train/valid.cache' images and labels... 20 found, 0 missing, 0 empty, 0 corrupt: 100% 20/20 [00:00<?, ?it/s]
Plotting labels to runs/train/Model/labels.jpg... 

AutoAnchor: 6.02 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/Model
Starting training for 3000 epochs...

 Epoch   gpu_mem       box       obj       cls    labels  img_size
    0/2999     13.4G    0.1137   0.06378    0.0494        77       640: 100% 4/4 [00:08<00:00,  2.05s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.21it/s]
                 all         20          0          0          0          0          0

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    1/2999     13.4G    0.1139   0.06479   0.04928        85       640: 100% 4/4 [00:03<00:00,  1.05it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.07it/s]
                 all         20          0          0          0          0          0
                 
```

## Test
![alt text](https://github.com/SurawutSukkum/Python_YOLOV5_Basler_Opencv/blob/main/Capture.JPG?raw=true)
