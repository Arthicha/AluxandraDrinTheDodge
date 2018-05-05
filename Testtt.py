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

CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[70,639],[105,221],[301,228],[360,639]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])

CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[335,193],[538,182],[587,638],[294,639]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])

cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)

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
                                                                                FILENAME='picture\\testpic\TestBottomRightSide.jpg')
    cv2.imshow('org',org)
    for p in range(0,len(plate)):
        plate[p] = Zkele(plate[p],method='3d')
        print(sorted_plate_pos[p])
        print(sorted_plate_orientation[p])
        cv2.imshow('plate',plate[p])
        cv2.waitKey(200)

    cam4.close()
    cv2.waitKey(0)
    cam4 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT)
    org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
                                                                                 plateOrientation=True, show=True,
                                                                                 LOAD_IMAGE=True,
                                                                                 FILENAME='picture\\testpic\TestBottomLeftallSide.jpg')
    cv2.imshow('org', org)
    for p in range(0, len(plate)):
        plate[p] = Zkele(plate[p], method='3d')
        print(sorted_plate_pos[p])
        print(sorted_plate_orientation[p])
        cv2.imshow('plate', plate[p])
        cv2.waitKey(200)

    cam4.close()
    cv2.waitKey(0)
    cam4 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)

    org, plate, sorted_plate_pos, sorted_plate_orientation = cam4.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
                                                                                 plateOrientation=True, show=True,
                                                                                 LOAD_IMAGE=True,
                                                                                 FILENAME='picture\\testpic\TestRightSide.jpg')

    cv2.waitKey(0)
    cv2.imshow('org', org)
    for p in range(0, len(plate)):
        plate[p] = Zkele(plate[p], method='3d')
        print(sorted_plate_pos[p])
        print(sorted_plate_orientation[p])
        cv2.imshow('plate', plate[p])
        cv2.waitKey(200)
    cv2.waitKey(0)
    cam4.close()
    cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                               CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
                               CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)
    IP.filter_plate()
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
    # ''' my code '''
    # # 335 193
    # # 538 182
    # # 587 638
    # # 294 639
    # # [[335,193],[538,182],[587,638],[294,639]]
    # # point1 = calculate_position([104,193],M)
    # # point2=calculate_position([80,479],M)
    # # print(point1)
    # # print(point2)
    # # [point1,point2,(81,420),(639,420)
    # # Right Side
    # # 93  81
    # # 116 450
    # # 523 560
    # # 559 81
    # capture, matrice = IP.four_point_transform(img, FOUR_POINTS, True)
    #
    # cv2.imshow('image', capture)
    # k = cv2.waitKey(0)
    # if k == ord('s'):
    #     cv2.imwrite(SAVE_IMG_NAME, capture)
