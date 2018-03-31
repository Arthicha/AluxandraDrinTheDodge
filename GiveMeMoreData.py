__author__ = ['Zumo Arthicha Srisuchinnawong']
__version__ = 2.0
__description__ = 'Data Generaing Program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import os
from os import listdir
import random
import copy

import cv2
import numpy as np

from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
from module.RandomFunction import *

# set printing resolution of numpy module
np.set_printoptions(threshold=np.inf)


MAIN_PATH = os.getcwd()

Font_Path = ["\\font\\ENGFONT\\","\\font\\ENGFONT\\","\\font\\THFONT\\"]
Word = ["NUM","EN","TH"]




Save_Path = "D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\"
Font_Size = 32
AUGMOUNT = 50
Image_Shape = (64, 32)
MAGNIFY = [90,110]
MORPH = [1,5]
MOVE = [-3,3]
GAMMA = [10,40]



wordlist = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", "ศูนย์ ", "หนึ่ง ", "สอง ", "สาม ", "สี่ ", "ห้า ", "หก ", "เจ็ด ", "แปด ",
            "เก้า "]
filename = {"zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
            "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "0": "0", "1": "1", "2": "2",
            "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "ศูนย์ ": "ZeroTH",
            "หนึ่ง ": "OneTH", "สอง ": "TwoTH", "สาม ": "ThreeTH", "สี่ ": "FourTH", "ห้า ": "FiveTH",
            "หก ": "SixTH",
            "เจ็ด ": "SevenTH", "แปด ": "EightTH", "เก้า ": "NineTH"}


cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = RND_MAGNIFY(gray,MAGNIFY=[90.0,110.0])
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

