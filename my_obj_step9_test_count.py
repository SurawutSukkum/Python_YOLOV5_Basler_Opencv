import cv2
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml') # ไฟล์ YOLO Model ของเรา

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    
    img_out = img.copy()
    
    key = cv2.waitKey(1)
    if (key == ord('q')) or (ret == False):
        break
    
    pred_image, obj_box = yolo.predictions(img)
    
    obj = 'bottle'
    count = 0
    
    for xmin,ymin,xmax,ymax,obj_name,obj_conf in obj_box:
        
        if obj_name == obj:
            cv2.rectangle(img_out,(xmin,ymin),(xmax,ymax),(0,255,255),2)
            count = count + 1
            
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img_out,obj + ' : ' + str(count),(20,70),font,4,(200,55,0),3)

    cv2.imshow('pred_image',img_out) 
        
cv2.destroyAllWindows()
cap.release()
