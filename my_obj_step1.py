import cv2

cap = cv2.VideoCapture(0)

count = 1 # ชื่อภาพเริ่มต้น 1.jpg
play = 0
image_folder = 'my_data_images' # ชื่อโฟลเดอร์ที่เก็บภาพ Data Set ของเรา

while(cap.isOpened()):
    
    ret, img = cap.read()
    img = cv2.flip(img,1)    

    cv2.imshow("img",img)

    key = cv2.waitKey(1)
    
    if  key == ord('s'): # กดปุ่ม s หนึ่งครั้ง จะเป็นการถ่ายภาพหนึ่งภาพ
        filesave = './' + image_folder + '/' + str(count)+".jpg"
        cv2.imwrite(filesave,img) # บันทึกภาพนิ่งที่ได้จากการถ่ายภาพ
        print(filesave)
        count = count + 1

    if key == ord('q'): # กดปุ่ม q เพื่อออกจากการถ่ายภาพ
        break
    
cap.release()
cv2.destroyAllWindows()


