# 1. system module
import os
import sys
import copy
import itertools

# 2. machine learning module
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# 3. mathematical module
import numpy as np
import pandas as pd
import math
import random

# 4. our own module
from module.Tenzor import TenzorCNN, TenzorNN, TenzorAE
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella
from module.RandomFunction import *
from module.Zkeleton import Zkele
from module.ManipulateTaDa import getData

''' add my code'''
from module import PaZum
# 5. visualization module
import matplotlib.pyplot as plt
from module.UniqueCameraClass import *
# 6. image processing module
import cv2

IMG_PATH = 'picture\\testpic\\BottomLeftLeftSide.jpg'
ROTATION = 90
SAVE_IMG_NAME = 'BL_Bside.jpg'
FOUR_POINTS = np.array([[557, 202], [377, 197], [289, 595], [559, 638]])
IMAGE_SIZE = (32,64)
CAMERA_ALL_OFFSET_Z = 25

CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 0
CAM_RIGHT_FOUR_POINTS = np.array([[93,81],[116,450],[523,560],[559,81]])



CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = 0
CAM_BOTTOM_MIDDLE_FOUR_POINTS =  np.array([[20, 512], [636, 510], [488, 305], [182, 309]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.03
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 175
CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL = 175
CAM_BOTTOM_MIDDLE_BOUNDARY = 10
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = -1
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -20
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = -305

CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = 0
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[106, 167], [301, 182], [349, 619], [68, 629]])
# np.array([[119, 173], [51, 639], [358, 638], [314, 189]])
# np.array([[302,237],[106,230],[65,638],[358,638]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])
CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
CAM_BOTTOM_RIGHT_BOUNDARY = 20
CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68#-50
CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167#-100

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 0
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[353, 153], [277, 588], [556, 613], [553, 143]])
# np.array([[544, 173], [553, 638], [269, 621], [348, 182]])
# np.array([[342,145],[267,628],[554,639],[550,143]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])
CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_LEFT_THRESH_KERNEL = 37
CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL = 47
CAM_BOTTOM_LEFT_BOUNDARY = 20
CAM_BOTTOM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -277#-300
CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -143#-100

cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                               CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
                               CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT,thresh_kernel=CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                             thresh_kernel_original=CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_RIGHT_MINIMUM_AREA
                             ,minimum_area_original=CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                             maximum_area_original=CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_RIGHT_LENGTH_PERCENT,
                               Offset_homo_x=CAM_BOTTOM_RIGHT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y)

def m_click(event, x, y, k, l):
    if (event == cv2.EVENT_LBUTTONUP):
        print(x, y)


def calculate_position(position, HomoMatrix):
    ''' for 2 dimen only'''
    new_position = (position[0], position[1], 1)
    new_position = np.matmul(HomoMatrix, np.reshape(new_position, (-1, 1)))
    new_position = np.reshape(new_position, (1, -1)).tolist()
    new_position = tuple(new_position[0][0:2])
    return new_position


while 1:
    org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
                                                                                plateOrientation=True, show=True,
                                                                                LOAD_IMAGE=True,
                                                                                FILENAME='Br_test.png')
    cv2.imshow('org',org)
    for p in range(0,len(plate)):
        plate[p] = Zkele(plate[p],method='3d')
        print(sorted_plate_pos[p])
        print(sorted_plate_orientation[p])
        cv2.imshow('plate',plate[p])
        cv2.waitKey(200)

    cam4.close()
    cv2.waitKey(0)
    # cam4 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
    #                        CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT,thresh_kernel=CAM_BOTTOM_LEFT_THRESH_KERNEL,
    #                          thresh_kernel_original=CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_LEFT_MINIMUM_AREA
    #                          ,minimum_area_original=CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_LEFT_MAXIMUM_AREA,
    #                          maximum_area_original=CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_LEFT_LENGTH_PERCENT,
    #                           Offset_homo_x=CAM_BOTTOM_LEFT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_LEFT_OFFSET_HOMO_Y)
    # org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
    #                                                                              plateOrientation=True, show=True,
    #                                                                              LOAD_IMAGE=True,
    #                                                                              FILENAME='Bl_test.png')
    # cv2.imshow('org', org)
    # for p in range(0, len(plate)):
    #     plate[p] = Zkele(plate[p], method='3d')
    #     print(sorted_plate_pos[p])
    #     print(sorted_plate_orientation[p])
    #     cv2.imshow('plate', plate[p])
    #     cv2.waitKey(0)
    #
    # cam4.close()
    # cv2.waitKey(0)
    # cam4 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)
    #
    # org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
    #                                                                              plateOrientation=True, show=True,
    #                                                                              LOAD_IMAGE=True,
    #                                                                              FILENAME='picture\\testpic\TestRightSide.jpg')
    #
    # cv2.waitKey(0)
    # cv2.imshow('org', org)
    # for p in range(0, len(plate)):
    #     plate[p] = Zkele(plate[p], method='3d')
    #     print(sorted_plate_pos[p])
    #     print(sorted_plate_orientation[p])
    #     cv2.imshow('plate', plate[p])
    #     cv2.waitKey(200)
    # cv2.waitKey(0)

    # cam4 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
    #                         CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS,thresh_kernel=CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
    #                         thresh_kernel_original=CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_MIDDLE_MINIMUM_AREA
    #                         ,minimum_area_original=CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
    #                         maximum_area_original=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT,
    #                         lengthpercent_original=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL,word_boundary=CAM_BOTTOM_MIDDLE_BOUNDARY,binarize_method=CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)
    # org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
    #                                                                              plateOrientation=True, show=True,
    #                                                                              LOAD_IMAGE=True,
    #                                                                              FILENAME='Bm_test_edit.png')
    # cv2.imshow('org', org)
    # print(len(plate))
    # for p in range(0, len(plate)):
    #     plate[p] = Zkele(plate[p], method='3d')
    #     print(sorted_plate_pos[p])
    #     print(sorted_plate_orientation[p])
    #     cv2.imshow('plate', plate[p])
    #     cv2.waitKey(200)
    #
    # cam4.close()
    # cv2.waitKey(0)

    # cam4.close()
    cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                               CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
                               CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT,thresh_kernel=CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                             thresh_kernel_original=CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_RIGHT_MINIMUM_AREA
                             ,minimum_area_original=CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                             maximum_area_original=CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_RIGHT_LENGTH_PERCENT,
                               Offset_homo_x=CAM_BOTTOM_RIGHT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y)


    # IP.filter_plate()
    # img = cv2.imread(IMG_PATH)
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', m_click)
    # cv2.waitKey(0)
    # rows, cols = img.shape
    # maxima = max(img.shape)
    # print(maxima)
    # blank_image = np.ones((cols, cols), np.uint8) * 255
    # blank_image[int((maxima - rows) / 2):int(maxima - (maxima - rows) / 2),
    # int((maxima - cols) / 2):int(maxima - (maxima - cols) / 2)] = img
    # cv2.imshow('image', blank_image)
    # cv2.setMouseCallback('image', m_click)
    # cv2.waitKey(0)
    # new_image_rows, new_image_cols = blank_image.shape
    # # blank_image[:,:]
    # M = cv2.getRotationMatrix2D((maxima / 2, maxima / 2), ROTATION, 1)
    # img = cv2.warpAffine(blank_image, M, (cols, cols), borderValue=255)
    # print(img.shape)
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', m_click)
    # cv2.waitKey(0)
    # path = os.getcwd()
    ''' my code '''
    # 335 193
    # 538 182
    # 587 638
    # 294 639
    # [[335,193],[538,182],[587,638],[294,639]]
    # point1 = calculate_position([104,193],M)
    # point2=calculate_position([80,479],M)
    # print(point1)
    # print(point2)
    # [point1,point2,(81,420),(639,420)
    # Right Side
    # 93  81
    # 116 450
    # 523 560
    # 559 81
    # capture, matrice = IP.four_point_transform(img, FOUR_POINTS, True)
    #
    # cv2.imshow('image', capture)
    # k = cv2.waitKey(0)
    # if k == ord('s'):
    #     cv2.imwrite(SAVE_IMG_NAME, capture)
