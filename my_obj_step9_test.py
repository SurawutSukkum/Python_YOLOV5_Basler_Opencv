import cv2
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml') # ไฟล์ YOLO Model ของเรา

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)

    key = cv2.waitKey(1)
    if (key == ord('q')) or (ret == False):
        break

    pred_image, obj_box = yolo.predictions(img)

    cv2.imshow('pred_image',pred_image) 
        
cv2.destroyAllWindows()
cap.release()
