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
                 
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 550, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

651 epochs completed in 1.782 hours.
Optimizer stripped from runs/train/Model/weights/last.pt, 92.9MB
Optimizer stripped from runs/train/Model/weights/best.pt, 92.9MB

Validating runs/train/Model/weights/best.pt...
Fusing layers... 
YOLOv5l summary: 367 layers, 46129818 parameters, 0 gradients, 107.7 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         20        100      0.986       0.99       0.99      0.777
     Symbols_address         20         20      0.994          1      0.995      0.797
         Symbols_UDI         20         21      0.995      0.952      0.977      0.748
          Symbols_TH         20         19      0.947          1       0.99      0.767
        Symbols_temp         20         20      0.997          1      0.995      0.795
        Symbols_Humi         20         20      0.996          1      0.995      0.777
Results saved to runs/train/Model
export: data=data/coco128.yaml, weights=['runs/train/Model/weights/best.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['torchscript', 'onnx']
YOLOv5 🚀 v6.1-250-g6adc53b Python-3.7.13 torch-1.11.0+cu102 CPU

Fusing layers... 
YOLOv5l summary: 367 layers, 46129818 parameters, 0 gradients, 107.7 GFLOPs

PyTorch: starting from runs/train/Model/weights/best.pt with output shape (1, 25200, 10) (88.6 MB)

TorchScript: starting export with torch 1.11.0+cu102...
TorchScript: export success, saved as runs/train/Model/weights/best.torchscript (176.6 MB)

ONNX: starting export with onnx 1.12.0...
ONNX: export success, saved as runs/train/Model/weights/best.onnx (176.4 MB)

Export complete (22.13s)
Results saved to /content/drive/MyDrive/yolo_training/yolov5/runs/train/Model/weights
Detect:          python detect.py --weights runs/train/Model/weights/best.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/Model/weights/best.onnx')
Validate:        python val.py --weights runs/train/Model/weights/best.onnx
Visualize:       https://netron.app
Training Done
                 
```

## Test
![alt text](https://github.com/SurawutSukkum/Python_YOLOV5_Basler_Opencv/blob/main/Capture.JPG?raw=true)
