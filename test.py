import cv2
import pytesseract
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
from pytesseract import Output
import re
import numpy as np
from pyzbar.pyzbar import decode
from pylibdmtx import pylibdmtx
from matplotlib import pyplot as plt
import sys
from yolo_predictions import YOLO_Pred
import time
import sys
import threading
from ctypes import *
import os
import time
import logging
from datetime import datetime
import relay_modbus
import relay_boards
import math
import imutils

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml')
print("User Current Version:-", sys.version)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

global Symbols_address_result
global Symbols_TH_result
global Symbols_Humi_result
global Symbols_UDI_result
global Symbols_temp_result
global TextSN90_result
global TextSN901_result
global TextSN902_result
global TextSN_result
global BarcodeTextSN_result
global BarcodeTextREF_result
global AddressText_result
global MadeInThailand_result
global BarcodeSN_result
global BarcodeREF_result
global BarcodeDatamatrix_result
global DateCheck_result
global BarcodeREF90_result
global BarcodeSN90_result
global step
global Serial_version
# Adding custom options
# Required: Configure serial port, for example:
#   On Windows: 'COMx'
#   On Linux:   '/dev/ttyUSB0'
SERIAL_PORT = 'COM3'

# Optional: Configure board address with 6 DIP switches on the relay board
# Default address: 1
address = 1

# Optional: Give the relay board a name
board_name = 'Relay board kitchen'

print("User Current Version:-", sys.version)
print("start")
now = datetime.now()
timestamp = datetime.fromtimestamp(datetime.timestamp(now))
timestamp = timestamp.strftime("%Y-%m-%d-%H%M%S")
logging.basicConfig(filename= timestamp + '.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.warning('This will get logged to a file')
# current date and time

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)
# thresholding
def thresholding(image):
    return   cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)
# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)
# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
def detectLine(img_raw):
    try:
        # Convert the img to grayscale
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
        # Apply edge detection method on the image
        img_blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        edges = cv2.Canny(img_blur, threshold1=100, threshold2=200)
        #cv2.namedWindow('img_blur', cv2.WINDOW_NORMAL)
        #cv2.imshow('img_blur', img_blur)
        # This returns an array of r and theta values
        lines = cv2.HoughLines(img_blur, 1, np.pi / 180, 255)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        recta = cv2.minAreaRect(contours[0])
        center_x, center_y, angle2 = recta
        # The below for loop runs till r and theta values
        # are in the range of the 2d array


        for r_theta in lines[:2]: # line no 2
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))

            try:
                angle = math.atan((y2-y1)/(x2-x1))
                angle = (180/math.pi)*angle
            except:
                angle =0
                pass
            # grab the dimensions of the image and calculate the center of the
            # image
            cv2.line(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 20)

        return angle,angle2
    except:
        angle  =0
        angle2 =0
        return angle, angle2
        pass
def detectLabel(img_raw1):
    gray = cv2.cvtColor(img_raw1, cv2.COLOR_BGR2GRAY)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cv2.namedWindow('detectLabel', cv2.WINDOW_NORMAL)
    cv2.imshow("detectLabel",thresh)
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(cnt)

            ratio = float(w) / h
            if (w > 400):
                print("Number of w, h", w, h)

                if ratio >= 0.9 and ratio <= 1.1:
                    img_raw0 = cv2.drawContours(img_raw1, [cnt], -1, (0, 255, 255), 3)
                    return 0, img_raw1
                else:
                    points = [(p[0][0], p[0][1]) for p in approx]
                    #print("points", points)
                    #slope = (points[3][1] - points[0][1]) / (points[3][0] - points[0][0])

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    color = (0, 0, 0)
                    thickness = 2
                    img_raw0 = cv2.line(img_raw1, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0, 0, 255), 2)
                    img_raw0 = cv2.drawContours(img_raw1, [cnt], -1, (0, 255, 255), 3)
                    cv2.namedWindow('img_raw1', cv2.WINDOW_NORMAL)
                    cv2.imshow("img_raw1",img_raw1)
                    return 1, img_raw1

    else:
        return 1, img_raw1
# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
g_bExit = False
# 显示图像
def put_text(x,y,img_raw1,objName,template):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (0, 180, 0)
    thickness = 5
    img_raw1 = cv2.putText(img_raw1, f'{objName} = {template}', (x,y), font, fontScale, color,
                           thickness,
                           cv2.LINE_AA)
    return img_raw1
def imageType(x,y,w,h,image,type):
    crop_img = image[y:y + h, x:x + w]
    gray_img = get_grayscale(crop_img)
    thresh_img = thresholding(gray_img)
    #opening = opening(thresh_img)
    #canny = canny(thresh_img)

    if type == "gray":
       #gray_img=dilate(gray_img)
       #gray_img=dilate(gray_img)
       return  gray_img,x,y,w,h
    if type == "org":
       #crop_img=dilate(crop_img)
       return  crop_img,x,y,w,h
    if type == "thresh":
       #thresh_img=dilate(thresh_img)
       return  thresh_img,x,y,w,h
def text_read(image,img_raw1,x,y,w,h,objName,template,angle):
    if(angle == 90):
        img_raw2 = cv2.rotate(img_raw1, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_raw2=img_raw1

    custom_config1 = r' --psm 6 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$% ^&*()-.° '
    text = pytesseract.image_to_string(img_raw2, lang='eng', config=custom_config1)
    #text = re.split('\n |, | |,|\*|\n', text)
    result = 0
    cnt = 0
    print("result text", objName, text, template)
    logging.warning(str(objName) + "=" +str(text))

    for i in text:
        if (i.find(str(template)) != -1):
            #print(objName, template)
            result = 1
        cnt = cnt + 1


    if (text.find(str(template)) != -1):
        #print(objName, template)
        #image3 = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255,0), 3)
        return str(template), image
    else:
        #print(objName+"result=", result,text)
        #image3 = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0,255), 3)
        return 0, image
def BarcodeRead(image,img_raw1,x,y,w,h,objName,template,angle):
    if(angle == 90):
        img_raw2 = cv2.rotate(img_raw1, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_raw2 = img_raw1
    detectedBarcodes = decode(img_raw2)
    if not detectedBarcodes:
        return 0, img_raw2
    else:
        for barcode in detectedBarcodes:
            if barcode.data != "":
                #print("barcode read=",objName,barcode.data.decode())
                logging.warning(str(objName) + "=" + str(barcode.data.decode()))
                return str(barcode.data.decode()), img_raw2

def BarcodeREF90(image,img_raw1,ymin,xmin,objName, template,command):
    x_text= 1200
    y_text= 250
    # Barcode REF rotate_90
    y = ymin - 500
    x = xmin - 40
    h = 610
    w = 10
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", img_raw2)
    result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
    print("resultorg1=",objName,result)
    logging.warning('BarcodeREF90 =' + str(result))
    if (result != "") & (result != 0) :
        put_text(x_text,y_text,image,objName,result)
        return result, img_raw1, x, y, w, h
    else:
        x = x - 20
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "orgAdj", img_raw2)
        result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('BarcodeREF90 Adj1 =' + str(result))
        if (result != "") & (result != 0):
            img_raw1 = put_text(x_text, y_text, image, objName, result)
            return result, img_raw1, x, y, w, h
        else:
            x = x + 40
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "orgAdj2", img_raw2)
            result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('BarcodeREF90 Adj2 =' + str(result))
            if (result != "") & (result != 0):
                img_raw1 = put_text(x_text, y_text, image, objName, result)
                return result, img_raw1, x, y, w, h
            else:
                img_raw1 = put_text(x_text, y_text, image, objName, result)
                return 0, img_raw1, x, y, w, h
def BarcodeTextREF90(image,img_raw1,ymin,xmin,objName, template,command):
    # Barcode Text REF
    x_text = 1200
    y_text = 300
    y = ymin- 320
    x = xmin
    h = 320
    w = 60

    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('BarcodeTextREF90  =' + str(result))
    if (result == template):
        put_text(x_text,y_text,image,objName,result)
        return result, img_raw2, x, y, w, h
    else:
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "grey", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('BarcodeTextREF90  Adj1 =' + str(result))
        if (result == template):
            put_text(x_text, y_text, image, objName, result)
            return result, img_raw2, x, y, w, h
        else:
            y = ymin - 320-10
            x = xmin - 20
            h = 320 + 20
            w = 60 + 20
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" ""
            #cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('BarcodeTextREF90  Adj2 =' + str(result))
            if (result == template):
                put_text(x_text, y_text, image, objName, result)
                return result, img_raw2, x, y, w, h
            else:
                y = ymin - 320 - 20
                x = xmin + 20
                h = 320+20
                w = 60+20
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                #cv2.imshow(str(objName) + "orgadj", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('BarcodeTextREF90  Adj3 =' + str(result))
                print("result OCR3", objName, result, template)
                if result == str(template):
                    put_text(x_text, y_text, image, objName, result)
                    return result, img_raw2, x, y, w, h
                else:
                    return 0, img_raw2, x, y, w, h
def BarcodeSN90(image,img_raw1,ymin,xmin,objName, template,command):

    # Barcode REF rotate_90
    x_text = 1200
    y_text = 350
    y= ymin - 520
    x = xmin - 40
    h = 620
    w = 10
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", img_raw2)
    result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('BarcodeSN90 =' + str(result))

    if (result != "") & (result != 0):
        put_text(x_text,y_text,image,objName,result)
        return result, img_raw2, x, y, w, h
    else:
        x = x-20
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
        result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('BarcodeSN90 Adj1=' + str(result))

        if (result != "") & (result != 0):
            put_text(x_text, y_text, image, objName, result)
            return result, img_raw2, x, y, w, h
        else:
            x = x + 40
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
            result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('BarcodeSN90 Adj2=' + str(result))

            if (result != "") & (result != 0):
                put_text(x_text, y_text, image, objName, result)
                return result, img_raw2, x, y, w, h
            else:
                return 0, img_raw2, x, y, w, h
def BarcodeTextSN90(image,img_raw1,ymin,xmin,objName, template,command):
    # Barcode Text SN
    x_text = 1200
    y_text = 400
    y = ymin-400
    x = xmin
    h = 410
    w = 65
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('BarcodeTextSN90 =' + str(result))

    if (result == template):
        put_text(x_text,y_text,image,objName,result)
        return result, image, x, y, w, h
    else:
        x = x + 20
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "grey", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('BarcodeTextSN90 Adj1=' + str(result))

        if (result == template):
            put_text(x_text, y_text, image, objName, result)

            return result, image, x, y, w, h
        else:
            x = x - 40
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" ""
            #cv2.imshow(str(objName) + "grey", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('BarcodeTextSN90 Adj2=' + str(result))

            if (result == template):
                put_text(x_text, y_text, image, objName, result)
                return result, image, x, y, w, h
            else:
                y = ymin - 400 - 20
                x = xmin -20
                h = 410 + 20
                w = 65 +20
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                #cv2.imshow(str(objName) + "grey", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('BarcodeTextSN90 Adj3=' + str(result))

                if result == str(template):
                    put_text(x_text, y_text, image, objName, result)
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def BarcodeTextSN901(image,img_raw1,ymin,xmin,objName, template,command):
    # Barcode Text SN
    y = ymin -370
    x = xmin
    h = 390
    w = 80
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('BarcodeTextSN901 =' + str(result))

    if (result == template):
        return result, image, x, y, w, h
    else:
        x = x + 20
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "grey", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('BarcodeTextSN901 Adj1=' + str(result))

        if (result == template):
            return result, image, x, y, w, h
        else:
            x = x - 40
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" ""
            #cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('BarcodeTextSN901 Adj2=' + str(result))

            if (result == template):
                return result, image, x, y, w, h
            else:
                y = ymin -370 - 20
                x = xmin
                h = 410 + 20
                w = 65 + 20
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                #cv2.imshow(str(objName) + "orgadj", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('BarcodeTextSN901 Adj3=' + str(result))

                if result == str(template):
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h

def TextREF90(image,img_raw1,ymin,xmin,objName, template,command):
    # TextSN90

    y = ymin -505
    x = xmin -100
    h = 625
    w = 60
    #put_text(900, 1700, img_raw1, objName, template)
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", img_raw2)
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('TextREF90 =' + str(result))

    if result == str(template):
        return 1, img_raw2, x, y, w, h
    else:
        x = x + 10
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('TextREF90 Adj1=' + str(result))
        if result == str(template):
            return 1, image, x, y, w, h
        else:
            x = x -20
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            cv2.imshow(str(objName) + "grayADj", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('TextREF90 Adj2=' + str(result))
            if result == str(template):
                return 1, image, x, y, w, h
            else:
                y = ymin - 505 - 5
                x = xmin - 100 - 5
                h = 625
                w = 60
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
                cv2.imshow(str(objName) + "grayADj", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('TextREF90 Adj3=' + str(result))
                if result == str(template):
                    return 1, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def TextSN90(img,img_raw1,ymin,xmin,objName, template,command):

    # TextSN90
    y = ymin -520
    x= xmin -115
    h= 620
    w = 60
    image=img.copy()
    # put_text(900, 1700, img_raw1, objName, template)
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", image)
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('TextSN90 =' + str(result))

    if result == str(template):
        return 1, image, x, y, w, h
    else:
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('TextSN90 Adj1=' + str(result))
        #cv2.imshow(str(objName) + "gray", img_raw2)
        if result == str(template):
            return 1, image, x, y, w, h
        else:
            y = ymin - 505 + 5
            x = xmin - 100 + 5
            h = 620
            w = 60
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "gray", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('TextSN90 Adj2=' + str(result))

            if result == str(template):
                return 1, image, x, y, w, h
            else:
                y = ymin - 505-10
                x = xmin - 100-10
                h = 620
                w = 60
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('TextSN90 Adj3=' + str(result))

                if result == str(template):
                    return 1, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def TextSN901(image,img_raw1,ymin,xmin,objName, template,command):
    # TextSN90
    y = ymin - 500
    x= xmin-50
    h = 620
    w = 60
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
    logging.warning('TextSN901 =' + str(result))

    if result == str(template):
        return 1, img_raw2, x, y, w, h
    else:
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
        logging.warning('TextSN901 Adj1=' + str(result))

        if result == str(template):
            return 1, img_raw2, x, y, w, h
        else:
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
            cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
            logging.warning('TextSN901 Adj2=' + str(result))
            if result == str(template):
                return 1, img_raw2, x, y, w, h
            else:
                y = ymin  -500 - 10
                x = xmin -50 - 10
                h = 620
                w = 120

                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 90)
                logging.warning('TextSN901 Adj3=' + str(result))

                if result == str(template):
                    return 1, img_raw2, x, y, w, h
                else:
                    return 0, img_raw2, x, y, w, h

def TextSN(image,img_raw1,ymin,xmin,objName, template,command):

    y = ymin-850
    x = xmin -25
    h = 110
    w = 1300
    #put_text(900, 1700, img_raw1, objName, template)
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", img_raw2)
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template,0)
    print("result OCR1", objName, result, template)
    logging.warning('TextSN =' + str(result))

    if result == str(template):
        return 1, img_raw2, x, y, w, h
    else:
        y = ymin -850
        x = xmin - 25 + 10
        h = 625
        w = 60
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
        logging.warning('TextSN Adj1=' + str(result))

        if result == str(template):
            return 1, image, x, y, w, h
        else:
            y = ymin -850
            x = xmin -25-15
            h = 625
            w = 60
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            cv2.imshow(str(objName) + "grayADj", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
            logging.warning('TextSN Adj2=' + str(result))

            if result == str(template):
                return 1, image, x, y, w, h
            else:
                return 0, image, x, y, w, h

def BarcodeSN(image,img_raw1,ymin,xmin,objName, template,command):
    # BarcodeSN
    x_text = 10
    y_text = 250
    y = ymin - 140
    x = xmin
    h = 145
    w = 750
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    # cv2.imshow(str(objName)+"org", img_raw2)
    result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
    logging.warning('BarcodeSN =' + str(result))
    # print("resultorg=",objName,result)
    if (result != "") & (result != 0):
        put_text(x_text, y_text, image, objName, result)
        return result, image, x, y, w, h
    else:
        y = ymin - 140
        x = xmin - 10
        h = 145
        w = 750
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
        logging.warning('BarcodeSN Adj1=' + str(result))

        # print("resultgray=", objName, result)
        if (result != "") & (result != 0):
            put_text(x_text, y_text, image, objName, result)
            return result, image, x, y, w, h
        else:
            y = ymin - 140
            x = xmin + 20 +20
            h = 145
            w = 750
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
            logging.warning('BarcodeSN Adj2=' + str(result))

            # print("resultthresh=", objName, result)
            if (result != "") & (result != 0):
                put_text(x_text, y_text, image, objName, result)
                return result, image, x, y, w, h
            else:
                y = ymin - 140
                x = xmin + 20 -40
                h = 145
                w = 750
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
                cv2.imshow(str(objName) + "orgAdj", img_raw2)
                result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
                logging.warning('BarcodeSN Adj3=' + str(result))
                # print("resultthresh=", objName, result)
                if (result != "") & (result != 0):
                    put_text(x_text, y_text, image, objName, result)
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def BarcodeTextSN(image,img_raw1,ymin,xmin,objName, template,command):
    x_text = 10
    y_text = 300
    y = ymin-10
    x = xmin+90
    h = 80
    w = 400
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
    logging.warning('BarcodeTextSN =' + str(result))

    #cv2.imshow(str(objName) + "org", img_raw2)
    # print("result",result)
    if (result == template):
        put_text(x_text, y_text, image, objName, result)
        return result, image, x, y, w, h
    else:
        y = ymin-10
        x = xmin + 90+10
        h = 80
        w = 400
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
        logging.warning('BarcodeTextSN Adj1=' + str(result))

        if (result == template):
            put_text(x_text, y_text, image, objName, result)
            return result, image, x, y, w, h
        else:
            y = ymin-10
            x = xmin + 90-20
            h = 80
            w = 400
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
            logging.warning('BarcodeTextSN Adj2=' + str(result))
            if (result == template):
                put_text(x_text, y_text, image, objName, result)
                return result, image, x, y, w, h
            else:
                y = ymin-10
                x = xmin + 90+5
                h = 80
                w = 400
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
                #cv2.imshow(str(objName) + "orgAdj", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
                logging.warning('BarcodeTextSN Adj3=' + str(result))

                if (result == template):
                    put_text(10, 250, image, objName, result)
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def BarcodeREF(image,img_raw1,ymin,xmin,objName, template,command):
    # BarcodeSN
    x_text = 10
    y_text = 350
    y = ymin - 150
    x = xmin-10
    h= 100
    w = 790
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName)+"org", img_raw2)
    result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
    logging.warning('BarcodeREF =' + str(result))

    print("resultorg1=",objName,result)
    if (result != "") & (result != 0):
        put_text(x_text, y_text, image, objName, result)
        return result, img_raw2, x, y, w, h
    else:
        y = ymin - 150
        x = xmin - 10 -10
        h = 100
        w = 790
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
        logging.warning('BarcodeREF Adj1=' + str(result))

        #print("resultgray=", objName, result)
        if (result != "") & (result != 0):
            put_text(x_text, y_text, image, objName, result)
            return result, image, x, y, w, h
        else:
            y = ymin - 150
            x = xmin - 10+20
            h = 100
            w = 790
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
            logging.warning('BarcodeREF Adj2=' + str(result))

            #print("resultthresh=", objName, result)
            if (result != "") & (result != 0):
                put_text(x_text, y_text, image, objName, result)
                return result, image, x, y, w, h
            else:
                y = ymin - 150
                x = xmin - 10-5
                h = 100 +50
                w = 790 +50
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                cv2.imshow(str(objName) + "orgAdj", img_raw2)
                result, img_raw2 = BarcodeRead(image, img_raw2, x, y, w, h, objName, template, 0)
                logging.warning('BarcodeREF Adj3=' + str(result))
                # print("resultthresh=", objName, result)
                if (result != "") & (result != 0):
                    put_text(x_text, y_text, image, objName, result)
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h
def BarcodeTextREF(image,img_raw1,ymin,xmin,objName, template,command):
    x_text = 10
    y_text = 400
    y = ymin-30
    x= xmin+90
    h= 100
    w= 650
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
    logging.warning('BarcodeTextREF =' + str(result))

    #cv2.imshow(str(objName) + "org", img_raw2)
    # print("result",result)
    if (result == template):
        put_text(x_text, y_text, image, objName, result)
        return result, image, x, y, w, h
    else:
        x_text = 10
        y_text = 400
        y = ymin - 30
        x = xmin + 90 +20
        h = 100
        w = 650
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
        logging.warning('BarcodeTextREF Adj1=' + str(result))

        if (result == template):
            put_text(x_text, y_text, image, objName, result)
            return result, image, x, y, w, h
        else:
            x_text = 10
            y_text = 400
            y = ymin - 30
            x = xmin + 90 - 10
            h = 100
            w = 650
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "thresh", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
            logging.warning('BarcodeTextREF Adj2=' + str(result))

            if (result == template):
                put_text(x_text, y_text, image, objName, result)
                return result, image, x, y, w, h
            else:
                x_text = 10
                y_text = 400
                y = ymin - 30
                x = xmin + 90 - 20
                h = 100
                w = 650
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
                #cv2.imshow(str(objName) + "orgAdj", img_raw2)
                result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
                logging.warning('BarcodeTextREF Adj3=' + str(result))

                if (result == template):
                    put_text(x_text, y_text, image, objName, result)
                    return result, image, x, y, w, h
                else:
                    return 0, image, x, y, w, h

def addressSymbols(image,img_raw1,ymin,xmin, template,command):
    # address
    y = ymin - 110
    x = xmin + 130
    h = 270
    w= 700

    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 180, 0)
    thickness = 5
    custom_config1 = r'-c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$%^&*()-., --psm 6'
    text = pytesseract.image_to_string(img_raw2, config=custom_config1)
    t = text
    text = t.split("\n")


    if len(text) > 0:
        if (text[0].find('Respironics')!= -1) & (text[0].find('Inc')!= -1)  & (text[1].find('1001 Murry Ridge Lane')!= -1) & (text[2].find('Murrysville')!= -1) & (text[2].find('PA 15668 USA')!= -1):
            #text = t.split("\n")
            img_raw1 = cv2.putText(img_raw1, str(text[0]), (x + 10, y + 300), font, fontScale, color,
                                   thickness, cv2.LINE_AA)
            img_raw1 = cv2.putText(img_raw1, str(text[1]), (x + 10, y + 360), font, fontScale, color,
                                   thickness, cv2.LINE_AA)
            img_raw1 = cv2.putText(img_raw1, str(text[2]), (x + 10, y + 420), font, fontScale, color,
                                   thickness, cv2.LINE_AA)
            return 1,img_raw1, x, y, w, h
            # print(text)
        else:
            return 0, img_raw1, x, y, w, h
    else:
        return 0, img_raw1, x, y, w, h
def MadeInThailand(image,img_raw1,ymin,xmin,objName, template,command):
    # MadeInThailand
    y = ymin + 150
    x = xmin + 80
    h = 200
    w = 460
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
    #cv2.imshow(str(objName) + "org", img_raw2)
    result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
    print("result OCR1", objName, result, template)

    if result == str(template):
        return 1, img_raw2, x, y, w, h
    else:
        y = ymin + 150
        x = xmin + 80+10
        h = 200
        w = 460
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
        #cv2.imshow(str(objName) + "gray", img_raw2)
        result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
        if result == str(template):
            return 1, image, x, y, w, h
        else:
            y = ymin + 150
            x = xmin + 80 -10
            h = 200
            w = 460
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
            #cv2.imshow(str(objName) + "grayADj", img_raw2)
            result, img_raw2 = text_read(image, img_raw2, x, y, w, h, objName, template, 0)
            if result == str(template):
                return 1, image, x, y, w, h
            else:
                return 0, image, x, y, w, h

def BarcodeDatamatrix(image,img_raw1,ymin,xmin, template,command):
    # Barcode datamatrix
    y = ymin + 60
    x = xmin - 210
    h = 220
    w = 220
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    detectedBarcodes_datamatrix = pylibdmtx.decode(img_raw2)

    # If not detected then print the message
    if not detectedBarcodes_datamatrix:
        y = ymin + 70
        x = xmin - 200 + 10
        h = 220
        w = 200
        img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
        detectedBarcodes_datamatrix = pylibdmtx.decode(img_raw2)

        if not detectedBarcodes_datamatrix:
            img_raw1 = put_text(10, 100, img_raw1, "Datamatrix", "0")
            return 0, img_raw1, x, y, w, h
        else:
            for barcode in detectedBarcodes_datamatrix:
                if barcode.data != "":
                    img_raw1 = put_text(10, 100, img_raw1, "Datamatrix", barcode.data.decode())
                    logging.warning('BarcodeDatamatrix Adj1=' + str(barcode.data.decode()))

                    return 1, img_raw1, x, y, w, h
    else:
        for barcode in detectedBarcodes_datamatrix:
            if barcode.data != "":
                img_raw1 = put_text(10, 100, img_raw1, "Datamatrix", barcode.data.decode())
                logging.warning('BarcodeDatamatrix =' + str(barcode.data.decode()))
                return 1, img_raw1, x, y, w, h
def BarcodeTextDatamatrix(image,img_raw1,ymin,xmin, template,command):
    # Barcode datamatrix
    y = ymin + 80
    x = xmin + 10
    h = 200
    w = 470

    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
    custom_config1 = r'-c tessedit_char_whitelist= ABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$%^&*abcdefghijklmnopqrstuvwxyz-., --psm 7'
    text = pytesseract.image_to_string(img_raw2, lang='eng', config=custom_config1)
    t = text
    print("BarcodeTextDatamatrix",text)
    text = re.split('\n |, | |,|\*|\n', t)
    print("BarcodeTextDatamatrix",text)

    text1 = text[0].split('(01)')
    text2 = text[1].split('(21)')
    if (len(text)<=7):
        if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
            put_text(10, 150, img_raw1, "DatamatrixText", str(text[0])+ " " + text[1])
            return 1, img_raw1, x, y, w, h
        else:
            x = x + 20
            img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
            custom_config1 = r'-c tessedit_char_whitelist= ABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$%^&*abcdefghijklmnopqrstuvwxyz-., --psm 7'
            text = pytesseract.image_to_string(img_raw2, lang='eng', config=custom_config1)
            t = text
            text = re.split('\n |, | |,|\*|\n', t)
            text1 = text[0].split('(01)')
            text2 = text[1].split('(21)')
            if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                return 1, img_raw1, x, y, w, h
            else:
                y = ymin + 80 - 50
                x = xmin + 10 - 50
                h = 200 + 50
                w = 470 + 50
                img_raw2, x, y, w, h = imageType(x, y, w, h, image, "org")  # "org" "gray" "thresh"
                custom_config1 = r'-c tessedit_char_whitelist= ABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$%^&*abcdefghijklmnopqrstuvwxyz-., --psm 7'
                text = pytesseract.image_to_string(img_raw2, lang='eng', config=custom_config1)
                t = text
                text = re.split('\n |, | |,|\*|\n', t)
                print("BarcodeTextDatamatrix", text , len(text))
                text1 = text[0].split('(01)')
                text2 = text[1].split('(21)')
                if( len(text) == 2):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1) :
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    else:
                        return 0, img_raw1, x, y, w, h
                elif(len(text) == 3):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    if (text[1].find(str(template)) != -1) & (text[2].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[1]) + " " + text[2])
                        return 1, img_raw1, x, y, w, h
                    else:
                        return 0, img_raw1, x, y, w, h
                elif(len(text) == 4):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    if (text[1].find(str(template)) != -1) & (text[2].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[1]) + " " + text[2])
                        return 1, img_raw1, x, y, w, h
                    if (text[2].find(str(template)) != -1) & (text[3].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[2]) + " " + text[3])
                        return 1, img_raw1, x, y, w, h
                    else:
                        return 0, img_raw1, x, y, w, h
                elif(len(text) == 5):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    if (text[1].find(str(template)) != -1) & (text[2].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[1]) + " " + text[2])
                        return 1, img_raw1, x, y, w, h
                    if (text[2].find(str(template)) != -1) & (text[3].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[2]) + " " + text[3])
                        return 1, img_raw1, x, y, w, h
                    if (text[3].find(str(template)) != -1) & (text[4].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[3]) + " " + text[4])
                        return 1, img_raw1, x, y, w, h
                    else:
                        return 0, img_raw1, x, y, w, h
                elif(len(text) == 6):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    if (text[1].find(str(template)) != -1) & (text[2].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[1]) + " " + text[2])
                        return 1, img_raw1, x, y, w, h
                    if (text[2].find(str(template)) != -1) & (text[3].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[2]) + " " + text[3])
                        return 1, img_raw1, x, y, w, h
                    if (text[3].find(str(template)) != -1) & (text[4].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[3]) + " " + text[4])
                        return 1, img_raw1, x, y, w, h
                    if (text[4].find(str(template)) != -1) & (text[5].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[4]) + " " + text[5])
                        return 1, img_raw1, x, y, w, h

                    else:
                        return 0, img_raw1, x, y, w, h

                elif(len(text) == 7):
                    if (text[0].find(str(template)) != -1) & (text[1].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[0]) + " " + text[1])
                        return 1, img_raw1, x, y, w, h
                    if (text[1].find(str(template)) != -1) & (text[2].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[1]) + " " + text[2])
                        return 1, img_raw1, x, y, w, h
                    if (text[2].find(str(template)) != -1) & (text[3].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[2]) + " " + text[3])
                        return 1, img_raw1, x, y, w, h
                    if (text[3].find(str(template)) != -1) & (text[4].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[3]) + " " + text[4])
                        return 1, img_raw1, x, y, w, h
                    if (text[4].find(str(template)) != -1) & (text[5].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[4]) + " " + text[5])
                        return 1, img_raw1, x, y, w, h
                    if (text[5].find(str(template)) != -1) & (text[6].find(str(template)) != -1):
                        put_text(10, 150, img_raw1, "DatamatrixText", str(text[5]) + " " + text[6])
                        return 1, img_raw1, x, y, w, h

                    else:
                        return 0, img_raw1, x, y, w, h
    else:
        return 0, img_raw1, x, y, w, h
def DateCheck(image,img_raw1,ymin,xmin, objName, template,command):
    # Date&time
    y = ymin + 100
    x = xmin - 100
    h = 100
    w = 350
    img_raw2, x, y, w, h = imageType(x, y, w, h, image, "gray")  # "org" "gray" "thresh"
    text = pytesseract.image_to_string(img_raw2, config=custom_config)
    t = text
    text = t.split("\n")
    if (t.find('2022') != -1):
        return 1, img_raw1, x, y, w, h
    else:
        return 0, img_raw1, x, y, w, h
def relay_control(relay,sts):
    # Create relay_modbus object
    _modbus = relay_modbus.Modbus(serial_port=SERIAL_PORT, verbose=False)

    # Open serial port
    try:
        _modbus.open()
    except relay_modbus.SerialOpenException as err:
        print(err)
        sys.exit(1)

    # Create relay board object
    board = relay_boards.R421A08(_modbus,
                                 address=address,
                                 board_name=board_name,
                                 verbose=False)
    if sts == 1:
        #print('Turn relay 1 on')
        board.on(relay)

    if sts == 0:
        #print('Turn relay 1 off')
        board.off(relay)

def monitor():
    print("BarcodeTextSN_result", BarcodeTextSN_result)
    print("Symbols_address_result", Symbols_address_result)
    print("Symbols_TH_result", Symbols_TH_result)
    print("Symbols_Humi_result", Symbols_Humi_result)
    print("Symbols_UDI_result", Symbols_UDI_result)
    print("Symbols_temp_result", Symbols_temp_result)
    print("TextSN_result", TextSN_result)
    print("BarcodeTextREF_result", BarcodeTextREF_result)
    print("AddressText_result", AddressText_result)
    print("MadeInThailand_result", MadeInThailand_result)
    print("BarcodeREF_result", BarcodeREF_result)
    print("BarcodeDatamatrix_result", BarcodeDatamatrix_result)
    print("DateCheck_result", DateCheck_result)
    print("TextSN90_result", TextSN90_result)
    print("TextSN901_result", TextSN901_result)
    print("BarcodeREF90_result", BarcodeREF90_result)
    print("TextREF90_result", TextREF90_result)
    print("BarcodeSN90_result", BarcodeSN90_result)
    print("BarcodeSN_result", BarcodeSN_result)
    print("BarcodeTextSN901_result", BarcodeTextSN901_result)

def image_show(image):
    newImage = image.copy()

    BarcodeTextSN_result = 0
    Symbols_address_result =0
    Symbols_TH_result = 0
    Symbols_Humi_result = 0
    Symbols_UDI_result = 0
    Symbols_temp_result = 0
    TextSN_result = 0
    BarcodeTextREF_result =0
    AddressText_result = 0
    MadeInThailand_result = 0
    BarcodeREF_result = 0
    BarcodeDatamatrix_result = 0
    DateCheck_result = 0
    TextSN90_result = 0
    TextSN901_result = 0
    BarcodeREF90_result = 0
    TextREF90_result = 0
    BarcodeSN90_result = 0
    BarcodeSN_result = 0
    BarcodeTextSN901_result = 0
    step = 0
    img_raw1 = cv2.rectangle(newImage, (0, 0), (2500, 450), (50, 0, 0), -3)

    while step < 9:
        pred_image, obj_box = yolo.predictions(newImage)
        print("Obj countstep",len(obj_box),step)
        if(len(obj_box) < 11):
            return img_raw1


        Serial_version = 'DS2 Postponement No Tank with Cell'
        for xmin, ymin, xmax, ymax, obj_name, obj_conf in obj_box:
            #print("Symbols x y", xmin, ymin,obj_name)
            if (obj_name == "Symbols_REF") & (step == 0):
                #print("Symbols_REF x y", xmin, ymin)
                if (xmin < 1700) & (xmin > 150) :
                    step = 1
                    print("step", step)
                    #print("Symbols_Symbols_REF x y", xmin, ymin)
                    BarcodeREF_result, img_raw1, x5, y5, w5, h5 = BarcodeREF(newImage, img_raw1, ymin, xmin, "BarcodeREF", '', '')
                    BarcodeTextREF_result, img_raw1, x6, y6, w6, h6 = BarcodeTextREF(newImage, img_raw1, ymin , xmin, "BarcodeREFText",BarcodeREF_result, '')

            if (BarcodeREF_result == 'PPX110F10CK'):
                Serial_version = 'DS2 Postponement No Tank with Cell'
            if (BarcodeREF_result == 'PPX120F10CK'):
                Serial_version = 'DS2 Advanced Postponement No Tank with Cell'
            if (BarcodeREF_result == 'PPX120F10K'):
                Serial_version = 'DS2 Advanced Postponement No Tank No Cell'

            if (obj_name == "Symbols_address") & (step == 1):
                #print("Symbols_address x y", xmin, ymin)
                if (xmin < 805) & (xmin > 10) & (ymin > 920) & (ymin < 1700):
                    step = 2
                    print("step", step)
                    TextSN_result,img_raw1, x1, y1, w1, h1  = TextSN(newImage,img_raw1, ymin, xmin,'TextSN', Serial_version,'')
                    AddressText_result,img_raw1, x2, y2, w2, h2 = addressSymbols(newImage,img_raw1, ymin, xmin, '','')
                    for xmin, ymin, xmax, ymax, obj_name, obj_conf in obj_box:
                        if obj_name == "Symbols_address":
                            x = xmin
                            y = ymin
                            w = 70
                            h = 76
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_address_result = 1
                        if obj_name == "Symbols_TH":
                            x = xmin
                            y = ymin
                            w = 70
                            h = 62
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_TH_result = 1
                        if obj_name == "Symbols_Humi":
                            x = xmin
                            y = ymin
                            w = 108
                            h = 82
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_Humi_result = 1
                        if obj_name == "Symbols_UDI":
                            x = xmin
                            y = ymin
                            w = 82
                            h = 50
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_UDI_result = 1
                        if obj_name == "Symbols_temp":
                            x = xmin
                            y = ymin
                            w = 126
                            h = 77
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_temp_result = 1
                        if obj_name == "Symbols_SN":
                            x = xmin
                            y = ymin
                            w = 50
                            h = 50
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_temp_result = 1
                        if obj_name == "Symbols_REF":
                            x = xmin
                            y = ymin
                            w = 50
                            h = 50
                            colors = (0, 255, 0)
                            cv2.rectangle(img_raw1, (x, y), (x + w, y + h), colors, 2)
                            Symbols_temp_result = 1

            if (obj_name == "Symbols_SN") & (step == 2):
                #print("Symbols_SN x y", xmin, ymin)
                if (xmin < 1700) & (xmin > 50) :
                    step = 3
                    print("step", step)
                    #print("Symbols_Symbols_SN x y", xmin, ymin)
                    BarcodeSN_result, img_raw1, x3, y3, w3, h3 = BarcodeSN(newImage, img_raw1, ymin, xmin, "BarcodeSN", '', '')
                    BarcodeTextSN_result, img_raw1, x4, y4, w4, h4 = BarcodeTextSN(newImage, img_raw1, ymin + 10, xmin, "BarcodeSNText",BarcodeSN_result, '')

            if (obj_name == "Symbols_TH") & (step == 3):
                #print("Symbols_TH x y", xmin, ymin)
                if (xmin < 1500) & (xmin > 50) :
                    step = 4
                    print("step", step)
                    #print("Symbols_TH x y", xmin, ymin)
                    DateCheck_result             ,img_raw1, x7, y7, w7, h7 = DateCheck(newImage,img_raw1, ymin, xmin, "Date", '','')
            if (obj_name == "Symbols_UDI")& (step == 4):
                #print("Symbols_UDI x y", xmin, ymin)
                if (xmin < 1500) & (xmin > 200) :
                    step = 5
                    print("step", step)
                    BarcodeDatamatrix_result     ,img_raw1, x8, y8, w8, h8 = BarcodeDatamatrix(newImage,img_raw1, ymin, xmin, '','')
                    BarcodeTextDatamatrix_result     ,img_raw1, x9, y9, w9, h9 =BarcodeTextDatamatrix(newImage,img_raw1, ymin, xmin,BarcodeDatamatrix_result,'')
            if (obj_name == "Symbols_temp")& (step == 5):
                #print("Symbols_Temp x y", xmin, ymin)
                if (xmin < 2000) & (xmin > 50):
                    step = 6
                    print("step", step)
                    MadeInThailand_result        ,img_raw1, x10, y10, w10, h10 = MadeInThailand(newImage,img_raw1, ymin, xmin,"MadeinThailand", 'Made in Thailand','')

            #angle90
            if (obj_name == "Symbols_SN") & (step == 6):
                if (xmin < 2300) & (xmin > 1900) & (ymin < 1900) & (ymin > 1700):
                    step = 7
                    print("step", step)
                    #print("Symbols_SN2 x y", xmin, ymin)
                    BarcodeSN90_result           , img_raw1, x11, y11, w11, h11 = BarcodeSN90(newImage, img_raw1, ymin, xmin,"BarcodeSN90", '', '')
                    BarcodeTextSN90_result       , img_raw1, x12, y12, w12, h12 = BarcodeTextSN90(newImage, img_raw1, ymin, xmin, "BarcodeTextSN90",BarcodeSN90_result, '')
                    TextSN90_result              , img_raw1, x13, y13, w13, h13 = TextSN90(newImage,img_raw1, ymin, xmin, "TextSN90" ,Serial_version,'')

            if (obj_name == "Symbols_REF") & (step == 7):
                if (xmin < 2300) & (xmin > 1900) & (BarcodeSN_result != "") & (BarcodeSN_result != 0):
                    step = 8
                    print("step", step)
                    #print("Symbols_REF2 x y", xmin, ymin)
                    BarcodeREF90_result          , img_raw1, x14, y14, w14, h14 = BarcodeREF90(newImage, img_raw1, ymin, xmin,"BarcodeREF90", '', '')
                    BarcodeTextREF90_result      , img_raw1, x15, y15, w15, h15 = BarcodeTextREF90(newImage, img_raw1, ymin, xmin, "BarcodeTextREF90",BarcodeREF90_result, '')
                    TextREF90_result             , img_raw1, x17, y17, w17, h17 = TextREF90(newImage,img_raw1, ymin, xmin, "TextREF90", Serial_version,'')

            if (obj_name == "Symbols_SN") & (step == 8):
                if (xmin < 2300) & (xmin > 1900) & (ymin < 1200) & (ymin > 900) &(BarcodeSN_result != "") & (BarcodeSN_result != 0):
                    step = 9
                    print("step", step)
                    #print("Symbols_REF2 x y", xmin, ymin)
                    BarcodeTextSN901_result      , img_raw1, x16, y16, w16, h16 = BarcodeTextSN901(newImage, img_raw1, ymin, xmin, "BarcodeTextSN901",BarcodeSN_result, '')
                    TextSN901_result             , img_raw1, x18, y18, w18, h18 = TextSN901(newImage,img_raw1, ymin, xmin, "TextSN901", Serial_version,'')
            #cv2.putText(img_raw1, f'{obj_name}', (xmin, ymin - 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 180, 0), 5)
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 8
            thickness = 10
            org5x = 1900
            org5y = 200
            org5 = (org5x, org5y)

            if (step == 9):
                step= 0
                if (TextSN_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)

                if (AddressText_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 3)

                if  (BarcodeSN_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 0), 3)

                if (BarcodeTextSN_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x4, y4), (x4 + w4, y4 + h4), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 3)

                if (BarcodeREF_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x5, y5), (x5 + w5, y5 + h5), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x5, y5), (x5 + w5, y5 + h5), (0, 255, 0), 3)

                if (BarcodeTextREF_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x6, y6), (x6 + w6, y6 + h6), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x6, y6), (x6 + w6, y6 + h6), (0, 255, 0), 3)

                if ( DateCheck_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x7, y7), (x7 + w7, y7 + h7), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x7, y7), (x7 + w7, y7 + h7), (0, 255, 0), 3)

                if (BarcodeDatamatrix_result  == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x8, y8), (x8 + w8, y8 + h8), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x8, y8), (x8 + w8, y8 + h8), (0, 255, 0), 3)

                if (BarcodeTextDatamatrix_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x9, y9), (x9 + w9, y9 + h9), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x9, y9), (x9 + w9, y9 + h9), (0, 255, 0), 3)

                if (MadeInThailand_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x10, y10), (x10 + w10, y10 + h10), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x10, y10), (x10 + w10, y10 + h10), (0, 255, 0), 3)

                if (BarcodeSN90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x11, y11), (x11 + w11, y11 + h11), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x11, y11), (x11 + w11, y11 + h11), (0, 255, 0), 3)

                if (BarcodeTextSN90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x12, y12), (x12 + w12, y12 + h12), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x12, y12), (x12 + w12, y12 + h12), (0,255, 0), 3)

                if (TextSN90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x13, y13), (x13 + w13, y13 + h13), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x13, y13), (x13 + w13, y13 + h13), (0, 255, 0), 3)

                if (BarcodeREF90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x14, y14), (x14 + w14, y14 + h14), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x14, y14), (x14 + w14, y14 + h14), (0, 255, 0), 3)

                if (BarcodeTextREF90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x15, y15), (x15 + w15, y15 + h15), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x15, y15), (x15 + w15, y15 + h15), (0, 255, 0), 3)

                if (BarcodeTextSN901_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x16, y16), (x16 + w16, y16 + h16), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x16, y16), (x16 + w16, y16 + h16), (0, 255, 0), 3)

                if (TextREF90_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x17, y17), (x17 + w17, y17 + h17), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x17, y17), (x17 + w17, y17 + h17), (0, 255, 0), 3)

                if (TextSN901_result == 0):
                    img_raw1 = cv2.rectangle(img_raw1, (x18, y18), (x18 + w18, y18 + h18), (0, 0, 255), 3)
                else:
                    img_raw1 = cv2.rectangle(img_raw1, (x18, y18), (x18 + w18, y18 + h18), (0, 255, 0), 3)

                if (BarcodeTextDatamatrix_result== 0) |(Symbols_address_result == 0) | (Symbols_TH_result == 0) | (Symbols_Humi_result == 0) | (Symbols_UDI_result == 0) | (Symbols_temp_result == 0)  | (TextSN_result == 0) | (BarcodeTextREF_result == 0) | (BarcodeSN_result == 0) |(BarcodeTextSN_result == 0) | (AddressText_result == 0)| (MadeInThailand_result == 0) | (BarcodeDatamatrix_result == 0) | (DateCheck_result == 0)| (TextSN90_result == 0) | (TextSN901_result == 0)  | (BarcodeTextSN901_result == 0) | (BarcodeSN90_result == 0)   | (TextREF90_result == 0) | (BarcodeDatamatrix_result == 0):
                    color = (0, 0, 255)

                    img_raw1 = cv2.rectangle(img_raw1, (org5x, org5y-200), (org5x+400, org5y+50), color, 5)
                    img_raw1 = cv2.putText(img_raw1, "NG", org5, font, fontScale, color, thickness, 3)
                    relay_control(3, 0)
                    relay_control(2, 1)

                    result  = "NG"
                    logging.warning('BarcodeTextSN_result ='+ str(BarcodeTextSN_result ))
                    logging.warning('Symbols_address_result =' + str(Symbols_address_result))
                    logging.warning('Symbols_TH_result =' + str(Symbols_TH_result))
                    logging.warning('Symbols_Humi_result =' +str(Symbols_Humi_result))
                    logging.warning('Symbols_UDI_result ='+ str(Symbols_UDI_result))
                    logging.warning('Symbols_temp_result ='+str(Symbols_temp_result))
                    logging.warning('TextSN_result ='+str(TextSN_result))
                    logging.warning('BarcodeTextREF_result ='+str(BarcodeTextREF_result))
                    logging.warning('AddressText_result ='+str(AddressText_result))
                    logging.warning('MadeInThailand_result ='+str(MadeInThailand_result))
                    logging.warning('BarcodeREF_result ='+str(BarcodeREF_result))
                    logging.warning('BarcodeDatamatrix_result ='+str(BarcodeDatamatrix_result))
                    logging.warning('DateCheck_result ='+str(DateCheck_result))
                    logging.warning('TextSN90_result ='+str(TextSN90_result))
                    logging.warning('TextSN901_result ='+str(TextSN901_result))
                    logging.warning('BarcodeREF90_result ='+str(BarcodeREF90_result))
                    logging.warning('TextREF90_result ='+str(TextREF90_result))
                    logging.warning('BarcodeSN90_result ='+str(BarcodeSN90_result))
                    logging.warning('BarcodeSN_result ='+str(BarcodeSN_result))
                    logging.warning('BarcodeTextSN901_result ='+str(BarcodeTextSN901_result))

                    logging.warning('Complete is ' + str(BarcodeSN_result))

                    return img_raw1,str(BarcodeSN_result),result
                else:
                    color = (0, 255, 0)
                    img_raw1 = cv2.rectangle(img_raw1, (org5x, org5y-200), (org5x+400, org5y+50), color, 5)
                    img_raw1 = cv2.putText(img_raw1, "OK", org5, font, fontScale, color, thickness, 3)
                    relay_control(3, 1)
                    relay_control(2, 0)
                    result = "OK"

                    logging.warning('BarcodeTextSN_result ='+ str(BarcodeTextSN_result ))
                    logging.warning('Symbols_address_result =' + str(Symbols_address_result))
                    logging.warning('Symbols_TH_result =' + str(Symbols_TH_result))
                    logging.warning('Symbols_Humi_result =' +str(Symbols_Humi_result))
                    logging.warning('Symbols_UDI_result ='+ str(Symbols_UDI_result))
                    logging.warning('Symbols_temp_result ='+str(Symbols_temp_result))
                    logging.warning('TextSN_result ='+str(TextSN_result))
                    logging.warning('BarcodeTextREF_result ='+str(BarcodeTextREF_result))
                    logging.warning('AddressText_result ='+str(AddressText_result))
                    logging.warning('MadeInThailand_result ='+str(MadeInThailand_result))
                    logging.warning('BarcodeREF_result ='+str(BarcodeREF_result))
                    logging.warning('BarcodeDatamatrix_result ='+str(BarcodeDatamatrix_result))
                    logging.warning('DateCheck_result ='+str(DateCheck_result))
                    logging.warning('TextSN90_result ='+str(TextSN90_result))
                    logging.warning('TextSN901_result ='+str(TextSN901_result))
                    logging.warning('BarcodeREF90_result ='+str(BarcodeREF90_result))
                    logging.warning('TextREF90_result ='+str(TextREF90_result))
                    logging.warning('BarcodeSN90_result ='+str(BarcodeSN90_result))
                    logging.warning('BarcodeSN_result ='+str(BarcodeSN_result))
                    logging.warning('BarcodeTextSN901_result ='+str(BarcodeTextSN901_result))
                    logging.warning('Complete is ' + str(BarcodeSN_result))

                    return img_raw1,str(BarcodeSN_result),result

'''  
        print(" ")
        print("BarcodeTextSN_result", BarcodeTextSN_result)
        print("Symbols_address_result", Symbols_address_result)
        print("Symbols_TH_result", Symbols_TH_result)
        print("Symbols_Humi_result", Symbols_Humi_result)
        print("Symbols_UDI_result", Symbols_UDI_result)
        print("Symbols_temp_result", Symbols_temp_result)
        print("TextSN_result", TextSN_result)
        print("BarcodeTextREF_result", BarcodeTextREF_result)
        print("AddressText_result", AddressText_result)
        print("MadeInThailand_result", MadeInThailand_result)
        print("BarcodeREF_result", BarcodeREF_result)
        print("BarcodeDatamatrix_result", BarcodeDatamatrix_result)
        print("DateCheck_result", DateCheck_result)
        print("TextSN90_result", TextSN90_result)
        print("TextSN901_result", TextSN901_result)
        print("BarcodeREF90_result", BarcodeREF90_result)
        print("TextREF90_result", TextREF90_result)
        print("BarcodeSN90_result", BarcodeSN90_result)
        print("BarcodeSN_result", BarcodeSN_result)
        print("BarcodeTextSN901_result", BarcodeTextSN901_result)
        print(" ")
'''



# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

previousTime = 0
currentTime = 0
relay_control(3, 0)
relay_control(2, 0)
cnt = 0
while camera.IsGrabbing():

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    custom_config = r'-c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ0123456789 tessedit_char_blacklist=?!@#$%^&*()-., --psm 6'
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img3 = image.GetArray()
        M = np.ones(img3.shape, dtype="uint8")
        img3 = cv2.add(img3, M)
        img4 = img3.copy()
        path = 'D:\pythonProjectOCRrev00\img1'
        cv2.imwrite(os.path.join(path, '54.jpg'), img4)
        # grab the dimensions of the image and calculate the center of the
        # image
        angle, angle2 = detectLine(img4)
        (h, w) = img4.shape[:2]
        (cX, cY) = (w , h )
        # rotate our image by -90 degrees around the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img4, M, (w, h))

        img4 = cv2.resize(img4, (600 * 2, 400 * 2), interpolation=cv2.INTER_AREA)
        cv2.putText(img4,  "Angle=" + str(angle) , (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.namedWindow("Rotated by Degrees", cv2.WINDOW_NORMAL)
        cv2.imshow("Rotated by Degrees", img4)

        #cv2.imshow('Raw', img3)
        #cv2.imwrite("image2.PNG", img3)
        pred_image, obj_box = yolo.predictions(rotated)
        print("Obj= ",len(obj_box),cnt)
        if(len(obj_box) >= 9):
            cnt = cnt + 1
            relay_control(3, 0)
            relay_control(2, 0)
        else:
            img_raw1 = rotated
            # Calculating the FPS
            currentTime = time.time()
            fps = (currentTime - previousTime)
            previousTime = currentTime
            #print(fps)
            cv2.putText(img_raw1, "Process Time = "+str(int(fps)) + " Second" + " cnt= "+str(cnt), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

            img_raw1 = cv2.resize(img_raw1, (600 * 2, 400 * 2), interpolation=cv2.INTER_AREA)
            cv2.namedWindow('HLA Label Inspection', cv2.WINDOW_NORMAL)
            cv2.imshow('HLA Label Inspection', img_raw1)
            cnt =0



        if cnt > 4:
            relay_control(3, 0)
            relay_control(2, 0)
            cnt=0

            img_raw1,BarcodeSN_result,result = image_show(rotated)
            print("TEST = " ,img_raw1,BarcodeSN_result,result)

            # Calculating the FPS
            currentTime = time.time()
            fps = (currentTime - previousTime)
            previousTime = currentTime
            # print(fps)
            cv2.putText(img_raw1, "Process Time = " + str(int(fps)) + " Second" + " cnt= " + str(cnt) + " Angle=" +  str(angle) , (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            img_raw1 = cv2.resize(img_raw1, (600 * 2, 400 * 2), interpolation=cv2.INTER_AREA)
            cv2.namedWindow('HLA Label Inspection', cv2.WINDOW_NORMAL)
            cv2.imshow('HLA Label Inspection', img_raw1)
            path = 'D:\pythonProjectOCRrev00\img'
            cv2.imwrite(os.path.join(path, str(BarcodeSN_result) + result + '.jpg'), img_raw1)
            logging.warning('Image Angle is ' + str(angle))
            logging.warning('Image Complete is ' + str(BarcodeSN_result) + result + '.jpg')
            print("success")

        if cnt > 2:
            img_raw1 = rotated
            # Calculating the FPS
            currentTime = time.time()
            fps = (currentTime - previousTime)
            previousTime = currentTime
            #print(fps)
            cv2.putText(img_raw1, "Process Time = "+str(int(fps)) + " Second" + " cnt= "+str(cnt) + " Angle=" +  str(angle) , (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)
            img_raw1 = cv2.resize(img_raw1, (600 * 2, 400 * 2), interpolation=cv2.INTER_AREA)
            cv2.namedWindow('HLA Label Inspection', cv2.WINDOW_NORMAL)
            cv2.imshow('HLA Label Inspection', img_raw1)

        if cnt < 1 & cnt > 0:
            time.sleep(1)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        grabResult.Release()

            # Releasing the resource
camera.StopGrabbing()



