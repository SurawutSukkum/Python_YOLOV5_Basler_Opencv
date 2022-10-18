import sys
import time

# Add system path to find relay_ Python packages
sys.path.append('.')
sys.path.append('..')

import relay_modbus
import relay_boards



# -- coding: utf-8 --
import sys
import threading
import msvcrt
from ctypes import *
import numpy as np
import time

import cv2
sys.path.append("C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")
#sys.path.append("../MvImport")
from MvCameraControl_class import *
import cv2
import numpy
import pytesseract
from pytesseract import Output
import re
import numpy as np
from pyzbar.pyzbar import decode
from pylibdmtx import pylibdmtx
from yolo_predictions import YOLO_Pred
import time
import sys
import threading


yolo = YOLO_Pred('my_obj7.onnx','my_obj7.yaml')
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


# Adding custom options


# Required: Configure serial port, for example:
#   On Windows: 'COMx'
#   On Linux:   '/dev/ttyUSB0'
SERIAL_PORT = 'COM11'

# Optional: Configure board address with 6 DIP switches on the relay board
# Default address: 1
address = 1

# Optional: Give the relay board a name
board_name = 'Relay board kitchen'


def print_relay_board_info(board):
    print('Relay board:')
    print('  Name:      {}'.format(board.board_name))
    print('  Type:      {}'.format(board.board_type))
    print('  Port:      {}'.format(board.serial_port))
    print('  Baudrate:  {}'.format(board.baudrate))
    print('  Addresses: {}'.format(board.num_addresses))
    print('  Relays:    {}'.format(board.num_relays))
    print('  Address:   {} (Configure DIP switches)'.format(board.address))
    print()


# Adding custom options
custom_config = r'--oem 3 --psm 6'


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


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

def detectLine(img_raw1):


    # Convert the img to grayscale
    gray = cv2.cvtColor(img_raw1, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh_img, 1, 2)
    #cv2.imshow("thresh_img",thresh_img)
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            print("Number of w, h", w, h)
            ratio = float(w) / h
            if (w > 800):
                if ratio >= 0.9 and ratio <= 1.1:
                    img_raw1 = cv2.drawContours(img_raw1, [cnt], -1, (0, 255, 255), 3)
                    return 0, img_raw1
                else:
                    img_raw1 = cv2.drawContours(img_raw1, [cnt], -1, (0, 255, 255), 3)
                    return 1, img_raw1

    else:
        return 0, img_raw1

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


g_bExit = False
# 显示图像
def TextSN90(image,img_raw1,ymin,xmin, template):

    # TextSN90
    y_TextSN90 = ymin - 300
    x_TextSN90 = xmin + 870
    h_TextSN90 = 230
    w_TextSN90 = 30
    crop_img_y_TextSN90 = ymin - 240
    crop_img_x_TextSN90 = xmin + 1845
    crop_img_h_TextSN90 = 650
    crop_img_w_TextSN90 = 45
    crop_img_TextSN90 = image[crop_img_y_TextSN90:crop_img_y_TextSN90 + crop_img_h_TextSN90,
                        crop_img_x_TextSN90:crop_img_x_TextSN90 + crop_img_w_TextSN90]
    # crop_img_TextSN90  = img_raw[y_TextSN90 :y_TextSN90  + h_TextSN90 ,x_TextSN90 :x_TextSN90  + w_TextSN90 ]
    crop_img_TextSN90 = cv2.rotate(crop_img_TextSN90, cv2.ROTATE_90_CLOCKWISE)
    gray_TextSN90 = get_grayscale(crop_img_TextSN90)
    thresh_TextSN90 = thresholding(gray_TextSN90)
    # opening = opening(gray)
    # canny = canny(gray)
    org_TextSN90 = (x_TextSN90 - 180, y_TextSN90 - 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    # org_TextSN90 = cv2.resize(org_TextSN90, (300 ,30), interpolation=cv2.INTER_AREA)

    cv2.imshow("crop_img_TextSN90", gray_TextSN90)
    text = pytesseract.image_to_string(gray_TextSN90, config=custom_config)
    t = text
    text = t.split("\n")
    #print(text)
    #print (len(text[0]))
    if ( text[0] == template):
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_TextSN90, font, fontScale, color, thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 255, 0), 3)
        return 1,img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 0, 255), 3)
        return 0,img_raw1
        # print(text)

def TextSN901(image,img_raw1,ymin,xmin, template):
    y_TextSN90 = ymin - 40
    x_TextSN90 = xmin + 870
    h_TextSN90 = 230
    w_TextSN90 = 30
    crop_img_y_TextSN90 = ymin + 420
    crop_img_x_TextSN90 = xmin + 1840
    crop_img_h_TextSN90 = 650
    crop_img_w_TextSN90 = 45
    crop_img_TextSN90 = image[crop_img_y_TextSN90:crop_img_y_TextSN90 + crop_img_h_TextSN90,
                        crop_img_x_TextSN90:crop_img_x_TextSN90 + crop_img_w_TextSN90]
    # crop_img_TextSN90  = img_raw[y_TextSN90 :y_TextSN90  + h_TextSN90 ,x_TextSN90 :x_TextSN90  + w_TextSN90 ]
    crop_img_TextSN90 = cv2.rotate(crop_img_TextSN90, cv2.ROTATE_90_CLOCKWISE)
    gray_TextSN90 = get_grayscale(crop_img_TextSN90)
    thresh_TextSN90 = thresholding(gray_TextSN90)
    # opening = opening(gray)
    # canny = canny(gray)
    org_TextSN90 = (x_TextSN90 - 180, y_TextSN90 - 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    # org_TextSN90 = cv2.resize(org_TextSN90, (300 ,30), interpolation=cv2.INTER_AREA)

    cv2.imshow("crop_img_TextSN901", thresh_TextSN90)
    text = pytesseract.image_to_string(thresh_TextSN90, config=custom_config)
    t = text
    text = t.split("\n")
    #print(text)
    # print (len(text[0]))
    if ( text[0] == template):
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_TextSN90, font, fontScale, color, thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 255, 0), 3)
        return 1,img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 0, 255), 3)
        return 0,img_raw1
        # print(text)

def TextSN902(image,img_raw1,ymin,xmin, template):
    y_TextSN90 = ymin - 300
    x_TextSN90 = xmin + 970
    h_TextSN90 = 230
    w_TextSN90 = 30
    crop_img_y_TextSN90 = ymin - 240
    crop_img_x_TextSN90 = xmin + 2035
    crop_img_h_TextSN90 = 650
    crop_img_w_TextSN90 = 45
    crop_img_TextSN90 = image[crop_img_y_TextSN90:crop_img_y_TextSN90 + crop_img_h_TextSN90,
                        crop_img_x_TextSN90:crop_img_x_TextSN90 + crop_img_w_TextSN90]
    # crop_img_TextSN90  = img_raw[y_TextSN90 :y_TextSN90  + h_TextSN90 ,x_TextSN90 :x_TextSN90  + w_TextSN90 ]
    crop_img_TextSN90 = cv2.rotate(crop_img_TextSN90, cv2.ROTATE_90_CLOCKWISE)
    gray_TextSN90 = get_grayscale(crop_img_TextSN90)
    thresh_TextSN90 = thresholding(gray_TextSN90)
    # opening = opening(gray)
    # canny = canny(gray)
    org_TextSN90 = (x_TextSN90 - 180, y_TextSN90 - 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    # org_TextSN90 = cv2.resize(org_TextSN90, (300 ,30), interpolation=cv2.INTER_AREA)

    cv2.imshow("crop_img_TextSN902", crop_img_TextSN90)
    text = pytesseract.image_to_string(crop_img_TextSN90, config=custom_config)
    t = text
    text = t.split("\n")
    # print(text)
    # print (len(text[0]))
    if ( text[0] == template):
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_TextSN90, font, fontScale, color, thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 255, 0), 3)
        return 1,img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_TextSN90, y_TextSN90), (x_TextSN90 + w_TextSN90, y_TextSN90 + h_TextSN90),
                                 (0, 0, 255), 3)
        return 0,img_raw1
        # print(text)

def TextSN(image,img_raw1,ymin,xmin, template):
    y_Symbols_address = ymin - 310
    x_Symbols_address = xmin - 25
    h_Symbols_address = 50
    w_Symbols_address = 600
    crop_img_Symbols_address = image[y_Symbols_address:y_Symbols_address + h_Symbols_address,
                               x_Symbols_address:x_Symbols_address + w_Symbols_address]
    gray_Symbols_address = get_grayscale(crop_img_Symbols_address)
    thresh_Symbols_address = thresholding(gray_Symbols_address)
    # opening = opening(gray)
    # canny = canny(gray)
    cv2.imshow("thresh_Symbols_address", gray_Symbols_address)
    org_Symbols_address = (x_Symbols_address, y_Symbols_address + 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(gray_Symbols_address, config=custom_config)
    t = text
    text = t.split("\n")
    #print(text)
    # print (len(text[0]))
    if (text[0] == template):
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_Symbols_address, font, fontScale, color, thickness,
                               cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_Symbols_address, y_Symbols_address),
                                 (x_Symbols_address + w_Symbols_address, y_Symbols_address + h_Symbols_address),
                                 (0, 255, 0), 3)
        return 1,img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_Symbols_address, y_Symbols_address),
                                 (x_Symbols_address + w_Symbols_address, y_Symbols_address + h_Symbols_address),
                                 (0, 0, 255), 3)
        return 0,img_raw1
        # print(text)

def BarcodeTextSN(image,img_raw1,ymin,xmin, template):
    # Barcode Text SN
    y_Barcode_Text = ymin - 310 + 100
    x_Barcode_Text = xmin - 30 + 550
    h_Barcode_Text = 35
    w_Barcode_Text = 300
    crop_img_Barcode_Text = image[y_Barcode_Text:y_Barcode_Text + h_Barcode_Text,
                            x_Barcode_Text:x_Barcode_Text + w_Barcode_Text]
    gray = get_grayscale(crop_img_Barcode_Text)
    thresh = thresholding(gray)
    # opening = opening(gray)
    # canny = canny(gray)
    cv2.imshow('gray_Barcode_Text_SN', gray)
    org_Barcode_Text = (x_Barcode_Text + 20, y_Barcode_Text + 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(gray, config=custom_config)
    t = text
    if len(text) > 8:
        text = t.split("\n")
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_Barcode_Text, font, fontScale, color, thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_Barcode_Text, y_Barcode_Text),
                                 (x_Barcode_Text + w_Barcode_Text, y_Barcode_Text + h_Barcode_Text), (0, 255, 0), 3)
        return 1, img_raw1
        # print(text)
    else:
        return 0,img_raw1
        img_raw1 = cv2.rectangle(img_raw1, (x_Barcode_Text, y_Barcode_Text),
                                 (x_Barcode_Text + w_Barcode_Text, y_Barcode_Text + h_Barcode_Text), (0, 0, 255), 3)

def BarcodeTextREF(image,img_raw1,ymin,xmin, template):
    # Barcode Text REF
    y_Barcode_Text_REF = ymin - 310 + 215
    x_Barcode_Text_REF = xmin - 30 + 550
    h_Barcode_Text_REF = 35
    w_Barcode_Text_REF = 200
    crop_img_Barcode_Text_REF = image[y_Barcode_Text_REF:y_Barcode_Text_REF + h_Barcode_Text_REF,
                                x_Barcode_Text_REF:x_Barcode_Text_REF + w_Barcode_Text_REF]
    gray_Barcode_Text_REF = get_grayscale(crop_img_Barcode_Text_REF)
    thresh_Barcode_Text_REF = thresholding(gray_Barcode_Text_REF)
    # opening = opening(gray)
    # canny = canny(gray)
    cv2.imshow('gray_Barcode_Text_REF', gray_Barcode_Text_REF)
    org_Barcode_Text_REF = (x_Barcode_Text_REF + 20, y_Barcode_Text_REF + 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(gray_Barcode_Text_REF, config=custom_config)
    t = text
    if len(text) > 8:
        text = t.split("\n")
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_Barcode_Text_REF, font, fontScale, color, thickness,
                               cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_Barcode_Text_REF, y_Barcode_Text_REF),
                                 (x_Barcode_Text_REF + w_Barcode_Text_REF, y_Barcode_Text_REF + h_Barcode_Text_REF),
                                 (0, 255, 0), 3)
        return 1,img_raw1
        # print(text)
    else:
        return 0,img_raw1
        img_raw1 = cv2.rectangle(img_raw1, (x_Barcode_Text_REF, y_Barcode_Text_REF),
                                 (x_Barcode_Text_REF + w_Barcode_Text_REF, y_Barcode_Text_REF + h_Barcode_Text_REF),
                                 (0, 0, 255), 3)

def addressSymbols(image,img_raw1,ymin,xmin, template):
    # address
    y_address = ymin - 310 + 280
    x_address = xmin - 20 + 75
    h_address = 170
    w_address = 350
    crop_img_address = image[y_address:y_address + h_address, x_address:x_address + w_address]
    # cv2.imshow('address', crop_img_address)
    # cv2.imshow('img_raw_OCR', img_raw_OCR)

    gray = get_grayscale(crop_img_address)
    thresh = thresholding(gray)
    # opening = opening(gray)
    # canny = canny(gray)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(crop_img_address, config=custom_config)
    t = text
    text = t.split("\n")
    # print(text)
    #print(text[0])
    #print(text[1])
    #print(text[2])
    if (text[0] == 'Respironics Inc.') & (text[1] == '1001 Murry Ridge Lane') & (text[2] == 'Murrysville, PA 15668 USA'):  # & (text[0] == 'Respironics Inc.') & (text[1] == '1001 Murry Ridge Lane') & (text[2] == 'Murrysville, PA 15668 USA') :
        text = t.split("\n")
        img_raw1 = cv2.putText(img_raw1, str(text[0]), (x_address + 10, y_address + 100), font, fontScale, color,
                               thickness, cv2.LINE_AA)
        img_raw1 = cv2.putText(img_raw1, str(text[1]), (x_address + 10, y_address + 130), font, fontScale, color,
                               thickness, cv2.LINE_AA)
        img_raw1 = cv2.putText(img_raw1, str(text[2]), (x_address + 10, y_address + 160), font, fontScale, color,
                               thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_address, y_address), (x_address + w_address, y_address + h_address),
                                 (0, 255, 0), 3)
        return 1,img_raw1
        # print(text)
    else:
        return 0,img_raw1
        img_raw1 = cv2.rectangle(img_raw1, (x_address, y_address), (x_address + w_address, y_address + h_address),
                                 (0, 0, 255), 3)

def MadeInThailand(image,img_raw1,ymin,xmin, template):
    # MadeInThailand
    y_MadeInThailand = ymin - 310 + 420
    x_MadeInThailand = xmin - 20 + 650
    h_MadeInThailand = 100
    w_MadeInThailand = 220
    crop_img_MadeInThailand = image[y_MadeInThailand:y_MadeInThailand + h_MadeInThailand,
                              x_MadeInThailand:x_MadeInThailand + w_MadeInThailand]
    org_MadeInThailand = (x_MadeInThailand + 10, y_MadeInThailand + 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(crop_img_MadeInThailand, config=custom_config)
    t = text
    text = t.split("\n")
    # print(text)
    # print (len(text[0]))
    if (text[0] == template):
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_MadeInThailand, font, fontScale, color, thickness,
                               cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_MadeInThailand, y_MadeInThailand),
                                 (x_MadeInThailand + w_MadeInThailand, y_MadeInThailand + h_MadeInThailand),
                                 (0, 255, 0), 3)
        return 1,img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_MadeInThailand, y_MadeInThailand),
                                 (x_MadeInThailand + w_MadeInThailand, y_MadeInThailand + h_MadeInThailand),
                                 (0, 0, 255), 3)
        return 0,img_raw1

def BarcodeSN(image,img_raw1,ymin,xmin, template):
    # BarcodeSN
    y_BarcodeSN = ymin - 310 + 40
    x_BarcodeSN = xmin - 20 + 495
    h_BarcodeSN = 60
    w_BarcodeSN = 380
    crop_BarcodeSN = image[y_BarcodeSN:y_BarcodeSN + h_BarcodeSN, x_BarcodeSN:x_BarcodeSN + w_BarcodeSN]
    org_BarcodeSN = (x_BarcodeSN + 50, y_BarcodeSN + 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    # Decode the barcode image
    gray_crop_BarcodeSN = get_grayscale(crop_BarcodeSN)
    thresh_crop_BarcodeSN = thresholding(gray_crop_BarcodeSN)
    # opening = opening(gray)
    # canny = canny(gray)
    cv2.imshow("gray_crop_BarcodeSN",gray_crop_BarcodeSN)
    detectedBarcodes_BarcodeSN = decode(gray_crop_BarcodeSN)
    # If not detected then print the message
    if not detectedBarcodes_BarcodeSN:
        # print("Barcode Not Detected or your barcode is blank/corrupted!")
        img_raw1 = cv2.rectangle(img_raw1, (x_BarcodeSN, y_BarcodeSN),
                                 (x_BarcodeSN + w_BarcodeSN, y_BarcodeSN + h_BarcodeSN), (0, 0, 255), 3)
        return 0,img_raw1
    else:

        # Traverse through all the detected barcodes in image
        img_raw1 = cv2.rectangle(img_raw1, (x_BarcodeSN, y_BarcodeSN),
                                 (x_BarcodeSN + w_BarcodeSN, y_BarcodeSN + h_BarcodeSN), (0, 255, 0), 3)

        for barcode in detectedBarcodes_BarcodeSN:
            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect
            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            if barcode.data != "":
                # Print the barcode data
                # print(barcode.data)
                img_raw1 = cv2.putText(img_raw1, str(barcode.data), org_BarcodeSN, font, fontScale, color, thickness,
                                       cv2.LINE_AA)

        return 1, img_raw1

def BarcodeDatamatrix(image,img_raw1,ymin,xmin, template):
    # Barcode datamatrix
    y_datamatrix = ymin - 310 + 150
    x_datamatrix = xmin - 20
    h_datamatrix = 100
    w_datamatrix = 100
    crop_img_datamatrix = image[y_datamatrix:y_datamatrix + h_datamatrix, x_datamatrix:x_datamatrix + w_datamatrix]
    org_datamatrix = (x_datamatrix, y_datamatrix - 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    # Decode the barcode image
    gray_datamatrix = get_grayscale(crop_img_datamatrix)
    thresh_datamatrix = thresholding(gray_datamatrix)
    # opening = opening(gray)
    # canny = canny(gray)
    detectedBarcodes_datamatrix = pylibdmtx.decode(gray_datamatrix)
    # If not detected then print the message
    if not detectedBarcodes_datamatrix:
        # print("Barcode Not Detected or your datamatrix")
        img_raw1 = cv2.rectangle(img_raw1, (x_datamatrix, y_datamatrix),
                                 (x_datamatrix + w_datamatrix, y_datamatrix + h_datamatrix), (0, 0, 255), 3)
        return 0, img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_datamatrix, y_datamatrix),
                                 (x_datamatrix + w_datamatrix, y_datamatrix + h_datamatrix), (0, 255, 0), 3)
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes_datamatrix:

            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect

            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            if barcode.data != "":
                # Print the barcode data
                # print(barcode.data)
                img_raw1 = cv2.putText(img_raw1, f'{barcode.data}', org_datamatrix, font, fontScale, color, thickness,
                                       cv2.LINE_AA)
        return 1, img_raw1

def DateCheck(image,img_raw1,ymin,xmin, template):
    # Date&time
    y_date = ymin - 310 + 195
    x_date = xmin - 20 + 330
    h_date = 50
    w_date = 150
    crop_img_date = image[y_date:y_date + h_date, x_date:x_date + w_date]
    org_date = (x_date + 10, y_date + 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 180, 0)
    thickness = 2
    text = pytesseract.image_to_string(crop_img_date, config=custom_config)
    t = text
    text = t.split("\n")
    # print(text)
    # print (len(text[0]))
    if len(text[0]) >= 10:
        img_raw1 = cv2.putText(img_raw1, str(text[0]), org_date, font, fontScale, color, thickness, cv2.LINE_AA)
        img_raw1 = cv2.rectangle(img_raw1, (x_date, y_date), (x_date + w_date, y_date + h_date), (0, 255, 0), 3)
        return 1, img_raw1
    else:
        img_raw1 = cv2.rectangle(img_raw1, (x_date, y_date), (x_date + w_date, y_date + h_date), (0, 0, 255), 3)
        return 0, img_raw1

def BarcodeREF90(image,img_raw1,ymin,xmin, template):

    # Barcode REF rotate_90
    y_Symbols_address_REF2 = ymin - 310 + 280-270
    x_Symbols_address_REF2 = xmin - 20 + 920
    h_Symbols_address_REF2 = 220
    w_Symbols_address_REF2 = 35
    crop_img_y_TextSN90 = ymin - 200
    crop_img_x_TextSN90 = xmin + 1850
    crop_img_h_TextSN90 = 650
    crop_img_w_TextSN90 = 120
    crop_img_Symbols_address_REF2 = image[crop_img_y_TextSN90:crop_img_y_TextSN90 + crop_img_h_TextSN90,
                                    crop_img_x_TextSN90:crop_img_x_TextSN90 + crop_img_w_TextSN90]
    org_Symbols_address_REF2 = (x_Symbols_address_REF2, y_Symbols_address_REF2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 180, 0)
    thickness = 2
    img_raw1 = cv2.rectangle(img_raw1, (x_Symbols_address_REF2, y_Symbols_address_REF2), (
    x_Symbols_address_REF2 + w_Symbols_address_REF2, y_Symbols_address_REF2 + h_Symbols_address_REF2), (0, 255, 0), 3)
    # Decode the barcode image
    #
    org_Symbols_address_REF2rotate_90 = cv2.rotate(crop_img_Symbols_address_REF2, cv2.ROTATE_90_CLOCKWISE)
    gray_Symbols_address_REF2 = get_grayscale(org_Symbols_address_REF2rotate_90)
    thresh_Symbols_address_REF2 = thresholding(gray_Symbols_address_REF2)
    # opening = opening(gray)
    # canny = canny(gray)

    #cv2.imshow("REF",org_Symbols_address_REF2rotate_90)
    detectedBarcodes_Symbols_address_REF2 = decode(org_Symbols_address_REF2rotate_90)
    # If not detected then print the message
    if not detectedBarcodes_Symbols_address_REF2:
        return 0, img_raw1
    else:
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes_Symbols_address_REF2:
            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect
            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            if barcode.data != "":
                # Print the barcode data
                # print(barcode.data)
                img_raw1 = cv2.putText(img_raw1, str(barcode.data), org_Symbols_address_REF2, font, fontScale, color,
                                       thickness, cv2.LINE_AA)
        return 1, img_raw1

def BarcodeSN90(image,img_raw1,ymin,xmin, template):

    # Barcode REF rotate_90
    y_Symbols_address_REF2 = ymin - 310 + 280
    x_Symbols_address_REF2 = xmin - 20 + 920
    h_Symbols_address_REF2 = 220
    w_Symbols_address_REF2 = 35
    crop_img_y_TextSN90 = ymin + 400
    crop_img_x_TextSN90 = xmin + 1850
    crop_img_h_TextSN90 = 650
    crop_img_w_TextSN90 = 120
    crop_img_Symbols_address_REF2 = image[crop_img_y_TextSN90:crop_img_y_TextSN90 + crop_img_h_TextSN90,
                                    crop_img_x_TextSN90:crop_img_x_TextSN90 + crop_img_w_TextSN90]
    org_Symbols_address_REF2 = (x_Symbols_address_REF2, y_Symbols_address_REF2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 180, 0)
    thickness = 2
    img_raw1 = cv2.rectangle(img_raw1, (x_Symbols_address_REF2, y_Symbols_address_REF2), (
    x_Symbols_address_REF2 + w_Symbols_address_REF2, y_Symbols_address_REF2 + h_Symbols_address_REF2), (0, 255, 0), 3)
    # Decode the barcode image
    #
    org_Symbols_address_REF2rotate_90 = cv2.rotate(crop_img_Symbols_address_REF2, cv2.ROTATE_90_CLOCKWISE)
    gray_Symbols_address_REF2 = get_grayscale(org_Symbols_address_REF2rotate_90)
    thresh_Symbols_address_REF2 = thresholding(gray_Symbols_address_REF2)
    # opening = opening(gray)
    # canny = canny(gray)

    #cv2.imshow("SN",org_Symbols_address_REF2rotate_90)
    detectedBarcodes_Symbols_address_REF2 = decode(org_Symbols_address_REF2rotate_90)
    # If not detected then print the message
    if not detectedBarcodes_Symbols_address_REF2:
        return 0, img_raw1
    else:
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes_Symbols_address_REF2:
            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect
            # Put the rectangle in image using
            # cv2 to heighlight the barcode
            if barcode.data != "":
                # Print the barcode data
                # print(barcode.data)
                img_raw1 = cv2.putText(img_raw1, str(barcode.data), org_Symbols_address_REF2, font, fontScale, color,
                                       thickness, cv2.LINE_AA)
        return 1, img_raw1

def image_show(image):

    Symbols_address_result = 0
    Symbols_TH_result   = 0
    Symbols_Humi_result = 0
    Symbols_UDI_result  = 0
    Symbols_temp_result = 0
    TextSN90_result  = 0
    TextSN901_result = 0
    TextSN902_result = 0
    TextSN_result = 0
    BarcodeTextSN_result = 0
    BarcodeTextREF_result = 0
    AddressText_result = 0
    MadeInThailand_result = 0
    BarcodeSN_result = 0
    BarcodeREF_result = 0
    BarcodeDatamatrix_result = 0
    DateCheck_result = 0
    BarcodeREF90_result = 0
    BarcodeSN90_result = 0

    img_raw      = cv2.resize(image, (600*2, 400*2), interpolation=cv2.INTER_AREA)
    img_raw_OCR  = cv2.resize(image, (600*2, 400*2), interpolation=cv2.INTER_AREA)
    img_raw1     = cv2.resize(image, (600*2, 400*2), interpolation=cv2.INTER_AREA)
    detect, img_raw1     = detectLine(img_raw1)
    #cv2.imshow("image1", img_raw1)
    #cv2.imwrite("image1.PNG", image)
    if detect == 1 :
        pred_image, obj_box = yolo.predictions(img_raw1)
        #cv2.imshow("image1", image)
        #cv2.imshow("img_raw1", img_raw)
        for xmin, ymin, xmax, ymax, obj_name, obj_conf in obj_box:
            cv2.putText(img_raw1, f'{obj_name}', (xmin, ymin - 40), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 180, 0), 2)
            #cv2.rectangle(img_raw1, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            #print(obj_name,ymin,xmin)
            #Symbols_UDI 212 110
            #Symbols_Humi 434 784
            #Symbols_TH 234 391
            #Symbols_temp 435 644
            #Symbols_address 388 43
            if obj_name == "Symbols_address":
                print(ymin, xmin)
                if (xmin < 205) & (xmin > 10) & (ymin > 200) & (ymin < 550):
                    # print("start")
                    #cv2.rectangle(img_raw1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    #cv2.putText(img_raw1, f'{obj_name}', (xmin, ymin - 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 180, 0), 2)

                    #TextSN90_result  ,img_raw1 = TextSN90(image,img_raw1, ymin, xmin, 'DS2 Advanced Postponement No Tank with Cell')
                    #TextSN901_result ,img_raw1 = TextSN901(image,img_raw1, ymin, xmin, 'DS2 Advanced Postponement No Tank with Cell')
                    #TextSN902_result ,img_raw1 = TextSN902(image,img_raw1, ymin, xmin, 'DS2 Advanced Postponement No Tank with Cell')
                    #BarcodeREF90_result         ,img_raw1 = BarcodeREF90(image,img_raw1, ymin, xmin-5, '')
                    #BarcodeSN90_result          ,img_raw1 = BarcodeSN90(image,img_raw1, ymin, xmin-5, '')
                    TextSN_result               ,img_raw1 = TextSN(img_raw,img_raw1, ymin, xmin, 'DS2 Advanced Postponement No Tank with Cell')
                    BarcodeTextSN_result        ,img_raw1 = BarcodeTextSN(img_raw,img_raw1, ymin, xmin, '')
                    BarcodeTextREF_result       ,img_raw1 = BarcodeTextREF(img_raw,img_raw1, ymin-10, xmin, '')
                    AddressText_result          ,img_raw1 = addressSymbols(img_raw,img_raw1, ymin, xmin, '')
                    MadeInThailand_result       ,img_raw1 = MadeInThailand(img_raw,img_raw1, ymin, xmin, 'Made in Thailand')
                    BarcodeSN_result            ,img_raw1 = BarcodeSN(img_raw,img_raw1, ymin, xmin, '')
                    BarcodeREF_result           ,img_raw1 = BarcodeSN(img_raw, img_raw1, ymin + 100, xmin , '')
                    BarcodeDatamatrix_result    ,img_raw1 = BarcodeDatamatrix(img_raw,img_raw1, ymin, xmin, '')
                    DateCheck_result            ,img_raw1 = DateCheck(img_raw,img_raw1, ymin, xmin, '')



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

                    '''       
                    print("Symbols_address_result ", Symbols_address_result)
                    print("Symbols_TH_result", Symbols_TH_result)
                    print("Symbols_Humi_result", Symbols_Humi_result)
                    print("Symbols_UDI_result", Symbols_UDI_result)
                    print("Symbols_temp_result", Symbols_temp_result)
                    print("TextSN90_result", TextSN90_result)
                    print("TextSN901_result", TextSN901_result)
                    print("TextSN902_result", TextSN902_result)
                    print("TextSN_result", TextSN_result)
                    print("BarcodeTextSN_result", BarcodeTextSN_result)
                    print("BarcodeTextREF_result", BarcodeTextREF_result)
                    print("AddressText_result", AddressText_result)
                    print("MadeInThailand_result", MadeInThailand_result)
                    print("BarcodeSN_result", BarcodeSN_result)
                    print("BarcodeREF_result", BarcodeREF_result)
                    print("BarcodeDatamatrix_result", BarcodeDatamatrix_result)
                    print("DateCheck_result", DateCheck_result)
                    print("BarcodeREF90_result", BarcodeREF90_result)
                    print("BarcodeSN90_result", BarcodeSN90_result)
                    '''
                    font = cv2.FONT_HERSHEY_COMPLEX
                    fontScale = 4
                    thickness = 10

                    org5x = 1000
                    org5y = 100
                    org5 = (org5x, org5y)

                    if (Symbols_address_result == 0) | (Symbols_TH_result == 0) | (Symbols_Humi_result == 0) | (Symbols_UDI_result == 0) | (Symbols_temp_result == 0)  | (TextSN_result == 0) | (BarcodeTextSN_result == 0)| (BarcodeTextREF_result == 0) | (AddressText_result == 0)| (MadeInThailand_result == 0) | (BarcodeSN_result == 0) | (BarcodeREF_result == 0) | (BarcodeDatamatrix_result == 0) | (DateCheck_result == 0): # | (TextSN90_result == 0) | (TextSN901_result == 0) | (TextSN902_result == 0)| (BarcodeREF90_result == 0) | (BarcodeSN90_result == 0):
                        color = (0, 0, 255)
                        img_raw1 = cv2.rectangle(img_raw1, (org5x, org5y-100), (org5x+200, org5y+20), color, 2)
                        img_raw1 = cv2.putText(img_raw1, "NG", org5, font, fontScale, color, thickness, 3)
                        relay_control(3, 0)


                    else:
                        color = (0, 255, 0)
                        img_raw1 = cv2.rectangle(img_raw1, (org5x, org5y-100), (org5x+200, org5y+20), color, 2)
                        img_raw1 = cv2.putText(img_raw1, "OK", org5, font, fontScale, color, thickness, 3)
                        relay_control(3, 1)
    else:
        relay_control(3, 0)

    cv2.imshow('HLA Label Inspection', img_raw1)
    k = cv2.waitKey(1) & 0xff

def Mono_numpy(self, data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
    data_mono_arr = data_.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 1], "uint8")
    numArray[:, :, 0] = data_mono_arr
    return numArray

def Color_numpy(self, data, nWidth, nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth * nHeight * 3:3]
    data_g = data_[1:nWidth * nHeight * 3:3]
    data_b = data_[2:nWidth * nHeight * 3:3]
    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3], "uint8")
    numArray[:, :, 0] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 2] = data_b_arr
    return numArray
# 判读图像格式是彩色还是黑白
def IsImageColor(enType):
    dates = {
        PixelType_Gvsp_RGB8_Packed: 'color',
        PixelType_Gvsp_BGR8_Packed: 'color',
        PixelType_Gvsp_YUV422_Packed: 'color',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
        PixelType_Gvsp_BayerGR8: 'color',
        PixelType_Gvsp_BayerRG8: 'color',
        PixelType_Gvsp_BayerGB8: 'color',
        PixelType_Gvsp_BayerBG8: 'color',
        PixelType_Gvsp_BayerGB10: 'color',
        PixelType_Gvsp_BayerGB10_Packed: 'color',
        PixelType_Gvsp_BayerBG10: 'color',
        PixelType_Gvsp_BayerBG10_Packed: 'color',
        PixelType_Gvsp_BayerRG10: 'color',
        PixelType_Gvsp_BayerRG10_Packed: 'color',
        PixelType_Gvsp_BayerGR10: 'color',
        PixelType_Gvsp_BayerGR10_Packed: 'color',
        PixelType_Gvsp_BayerGB12: 'color',
        PixelType_Gvsp_BayerGB12_Packed: 'color',
        PixelType_Gvsp_BayerBG12: 'color',
        PixelType_Gvsp_BayerBG12_Packed: 'color',
        PixelType_Gvsp_BayerRG12: 'color',
        PixelType_Gvsp_BayerRG12_Packed: 'color',
        PixelType_Gvsp_BayerGR12: 'color',
        PixelType_Gvsp_BayerGR12_Packed: 'color',
        PixelType_Gvsp_Mono8: 'mono',
        PixelType_Gvsp_Mono10: 'mono',
        PixelType_Gvsp_Mono10_Packed: 'mono',
        PixelType_Gvsp_Mono12: 'mono',
        PixelType_Gvsp_Mono12_Packed: 'mono'}
    return dates.get(enType, '未知')


# 需要显示的图像数据转换
def image_control(image,stFrameInfo):
    if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image_show(image=image)#显示
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
        image = imaga.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
        image = image.reshape((stFrameInfo.nHeight,stFrameInfo.nWidth))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
        image = image.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_show(image=image)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_YUV422_Packed:#YUV422_8_UYVY有问题
        image = image.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,2)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_Y422)
        image_show(image=image)
    else:
        print("Not support ImageFormat!!! \n")


#实现GetImagebuffer函数取流，HIK格式转换函数
def work_thread_1(cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    #print("work_thread_1!\n")
    img_buff = None
    previousTime = 0
    currentTime = 0
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            #print ("MV_CC_GetOneFrameTimeout: Width[%d], Height[%d], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            time_start = time.time()
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                #print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
            elif IsImageColor(stFrameInfo.enPixelType) == 'color':
                #print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight* 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                #print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                #print("convert ok!!")
                # 转OpenCV
                # 黑白处理
                if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff,count=int(stConvertParam.nDstLen), dtype=np.uint8)
                    img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    #print("mono ok!!")
                    image_show(image=img_buff)  # 显示图像函数
                # 彩色处理
                if IsImageColor(stFrameInfo.enPixelType) == 'color':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                    #print("color ok!!")
                    image_show(image=img_buff)  # 显示图像函数
                time_end = time.time()
                #print('time cos:', time_end - time_start, 's')
        else:
            print ("no data[0x%x]" % ret)
        if g_bExit == True:
                break

#实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
def work_thread_2(cam=0, pData=0, nDataSize=0):
    #global img_buff
    img_buff = None
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print ("MV_CC_GetImageBuffer: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
            elif IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # # 存raw图看看转化成功没有
                # file_path = "AfterConvert_RGB.raw"
                # file_open = open(file_path.encode('ascii'), 'wb+')
                # try:
                #     image_save= (c_ubyte * stConvertParam.nDstBufferSize)()
                #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                #     file_open.write(img_buff)
                #     print("raw ok!!")
                # except:
                #     raise Exception("save file executed failed:%s" % e.message)
                # finally:
                #     file_open.close()
            # 黑白处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff),stConvertParam.pDstBuffer,stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                img_buff = img_buff.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                print("mono ok!!")
                image_show(image=img_buff)  # 显示图像函数
            # 彩色处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)#data以流的形式读入转化成ndarray对象
                img_buff = img_buff.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth,3)
                print("color ok!!")
                image_show(image=img_buff)  # 显示图像函数
            else:
                print("no data[0x%x]" % ret)
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        if g_bExit == True:
            break

#实现getoneframe函数取流，OpenCV自带格式转换函数
def work_thread_3(cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = None
    print("work_thread_1!\n")
    #image = numpy.zeros((640, 480), dtype=numpy.uint8)
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            cdll.msvcrt.memcpy(byref(img_buff),pData, stFrameInfo.nFrameLen)
            img_buff = np.frombuffer(img_buff, count=int(stFrameInfo.nFrameLen), dtype=np.uint8) #data以流的形式读入转化成ndarray对象
            image_control(img_buff, stFrameInfo)
        if g_bExit == True:
            break

#回调函数
#实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
winfun_ctype = WINFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)
def image_callback(pData, pFrameInfo, pUser):
        #global img_buff
        img_buff = None
        stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
        if stFrameInfo:
            print ("callback:get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
            elif IsImageColor(stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
            # ---
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # 转OpenCV
                # 黑白处理
                if IsImageColor(stFrameInfo.enPixelType) == 'mono':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    print("mono ok!!")
                    image_show(image=img_buff)  # 显示图像函数
                # 彩色处理
                if IsImageColor(stFrameInfo.enPixelType) == 'color':
                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                    img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                    img_buff = img_buff.reshape(stFrameInfo.nHeight,stFrameInfo.nWidth,3)
                    print("color ok!!")
                    image_show(image=img_buff)  # 显示图像函数
CALL_BACK_FUN = FrameInfoCallBack(image_callback)


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



if __name__ == "__main__":

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
        sys.exit()

    print ("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("gige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("u3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("user serial number: %s" % strSerialNumber)

    nConnectionNum = '0'

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    
    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    ret = cam.MV_CC_SetIntValue("GevHeartbeatTimeout", 5000)
    if ret != 0:
        print("Warning: Set GevHeartbeatTimeout  fail! ret[0x%x]" % ret)

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
            if ret != 0:
                print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret =cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print ("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print ("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # # ch:注册抓图回调 | en:Register image callback
    # ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
    # if ret != 0:
    #     print("register image callback fail! ret[0x%x]" % ret)
    #     sys.exit()
    # else:
    #     print("Start callback grab!!! ")

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    # 设置 bufffer 大小
    data_buffer = (c_ubyte * nPayloadSize)()
    #单帧取流函数，主动取流方法实现
    try:
        #work_thread_1 #实现GetImagebuffer函数取流，HIK格式转换函数
        #work_thread_2 #实现MV_CC_GetImageBuffer函数取流，HIK格式转换函数
        #work_thread_3 #实现getoneframe函数取流，OpenCV自带格式转换函数
        hThreadHandle= threading.Thread(target=work_thread_1,args=(cam, byref(data_buffer), nPayloadSize))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")

    print ("press a key to stop grabbing.")
    msvcrt.getch()
    g_bExit = True

    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()
