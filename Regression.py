# import library
import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella
from module.UniqueCameraClass import *

# Measures are in milimeters

LOAD_IMAGE_NAME = "Br_tune_latest1.jpg"
#   'C:\\Users\\cha45\\PycharmProjects\\piv\\Ltest4.jpg'
# "L_tune.png"
#    'C:\\Users\\cha45\\PycharmProjects\\piv\\Ltest4.jpg'#newR_test.png
SAVE_IMAGE_NAME = 'Br_tune_latest1.jpg'
FROM_FILE = True
TEST_MODEL = True
CLASS = True
CAM_NAME = 'Br_bottom'
MODEL_FILE_NAME = 'Br_bottom'
CAM_ORIENT = 0
CAM_FOUR_POINT = np.array([[121, 162], [315, 174], [360, 621], [73, 619]])
# 38.5 from inside
# 31.5 from left
'''Regression Parameter'''
SAVED_MODEL_NAME = 'Br_bottom'
KERNEL_SIZE = (4, 4)
NUMBER_OF_POINTS = (15, 10)
DIFFERENCE_DISTANCE_PER_POINT = [30, 30]
SHIFT_X = 500-298
# -190
# 527.5
# 527.5-240 = 287.5
SHIFT_Y = 700-445
# 420
# 420+210 =630
BINARY_THRESHOLD = 70
# right
# from above = 57
# from inside = 20.75

# left
# from above = 54 cm
# from inside = 65 cm
'''Cam config for run with cam'''
CAMERA_ALL_OFFSET_Z = 25

CAM_BOTTOM_MIDDLE_NAME = 'Bm'
CAM_BOTTOM_MIDDLE_PORT = 1
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -180
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[0, 433], [136, 157], [500, 161], [639, 422]])
# np.array([[0, 514], [148, 347], [547, 330], [639, 477]])
# np.array([[17, 483], [178, 293], [485, 285], [637, 479]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.03
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 175
CAM_BOTTOM_MIDDLE_BOUNDARY = 10
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = -1
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -17
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = -285

cam_bottom_middle = Retinutella(CAM_BOTTOM_MIDDLE_NAME, CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION,
                                CAM_BOTTOM_MIDDLE_MODE, CAM_BOTTOM_MIDDLE_FOUR_POINTS, CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                                CAM_BOTTOM_MIDDLE_MINIMUM_AREA, CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                                CAM_BOTTOM_MIDDLE_LENGTH_PERCENT, CAM_BOTTOM_MIDDLE_BOUNDARY,
                                CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)

CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[70, 639], [105, 221], [301, 228], [360, 639]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])

CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[335, 193], [538, 182], [587, 638], [294, 639]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])

cam = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                          CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)

'''********************************************************'''


def Regression_HaHA(Image_naja, kernel_size=(4, 4), binarization_thresh_kernel_size=21, number_of_points=(12, 17),
                    difference_distance_per_point=[-120, -120], shift_x=2040, shift_y=1320, camera_name='left'):
    '''
    :param Image_naja: Image contain the point aka chessboard or whatever 
    :param kernel_size: kernel_size use to erode 
    :param number_of_points:  number of points in picture (rows,columns) 
    :param difference_distance_per_point:  point_to_point_distance(y,x)
    :param shift_x:     start point of x
    :param shift_y:     start point of y
    :return: 
    '''
    conWorldMatX = np.array([])
    conWorldMatY = np.array([])

    imgC = Image_naja
    imgBW = imgC
    # imgBW = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    # img = IP.binarize(imgBW, IP.SAUVOLA_THRESHOLDING, binarization_thresh_kernel_size)
    ret, img = cv2.threshold(imgBW, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    img = cv2.bitwise_not(img)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = np.array(img, dtype=np.uint8)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw contours and find centers
    for i in range(len(contours) - number_of_points[0] * number_of_points[1], len(contours)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cnt = contours[i]
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        conWorldMatX = np.append(conWorldMatX, [cx])
        conWorldMatY = np.append(conWorldMatY, [cy])
        mc = (cx, cy)
        cv2.circle(imgC, mc, 3, color, -1, 8, 0)
    conWorldMatX = np.reshape(conWorldMatX, (-1, number_of_points[1]))
    conWorldMatX = np.sort(conWorldMatX, axis=1)
    conWorldMatX = np.reshape(conWorldMatX, -1)
    print('-------------')
    print('center X:')
    print(conWorldMatX)
    print('center Y:')
    print(conWorldMatY)
    print('------------------')
    cv2.imshow('dot', imgC)
    cv2.waitKey(0)
    # *************************************************************************
    # REAL WORLD POINTS
    # *************************************************************************

    # X MATRIX
    RealWorldMatX = np.zeros(shape=number_of_points)
    RealWorldRolX = np.array([])
    for i in range(0, number_of_points[1]):
        if i in range(1, number_of_points[1]):
            RealWorldRolX = np.append(RealWorldRolX, [(shift_x) + (difference_distance_per_point[0] * i)])
        else:
            RealWorldRolX = np.append(RealWorldRolX, [shift_x])
    for j in range(0, number_of_points[0]):
        RealWorldMatX[j] = RealWorldRolX
    RealWorldMatX = RealWorldMatX.flatten()
    # RealWorldMatX = RealWorldMatX.reshape(-1,1)
    # print(len(RealWorldMatX))

    # Y MATRIX
    RealWorldMatY = np.zeros(shape=number_of_points)
    RealWorldRolY = np.zeros(shape=(1, number_of_points[1]))
    for l in range(0, number_of_points[0]):
        RealWorldRolY[:1] = (shift_y) + (difference_distance_per_point[1] * l)
        RealWorldMatY[l] = RealWorldRolY
        RealWorldRolY.fill(0)
    RealWorldMatY = RealWorldMatY.flatten()
    RealWorldMatY = RealWorldMatY.reshape(-1, 1)
    print('***************')
    print('Real world X :')
    print(RealWorldMatX)
    print('Real world Y :')
    print(RealWorldMatY)
    print('***************')
    DisX = np.zeros(shape=(number_of_points[0] * number_of_points[1], 6))
    DisY = np.zeros(shape=(number_of_points[0] * number_of_points[1], 6))
    for k in range(len(conWorldMatX)):
        # X_DATA
        X3 = np.power(conWorldMatX[k], 3)
        XY2 = (conWorldMatX[k]) * (np.power(conWorldMatY[k], 2))
        X5 = np.power(conWorldMatX[k], 5)
        X3Y2 = (np.power(conWorldMatX[k], 3)) * (np.power(conWorldMatY[k], 2))
        XY4 = conWorldMatX[k] * (np.power(conWorldMatY[k], 4))
        X2 = (np.power(conWorldMatX[k], 2))
        Y2 = (np.power(conWorldMatY[k], 2))
        XY = (conWorldMatX[k]) * (conWorldMatY[k])

        # #Y DATAS
        YX2 = (conWorldMatY[k]) * (np.power(conWorldMatX[k], 2))
        Y3 = np.power(conWorldMatY[k], 3)
        YX4 = (conWorldMatY[k]) * (np.power(conWorldMatX[k], 4))
        X2Y3 = (np.power(conWorldMatX[k], 2)) * (np.power(conWorldMatY[k], 3))
        Y5 = np.power(conWorldMatY[k], 5)

        '''my data'''
        X = conWorldMatX[k]
        Y = conWorldMatY[k]
        X_divide = 1 / (0.00001 + conWorldMatX[k])
        Y_divide = 1 / (0.00001 + conWorldMatY[k])
        XY_divide = X_divide * Y_divide

        dummyX = np.array([X, Y, XY2, X2, Y2, XY])  # X3,X5, X3Y2, XY4, , X_divide,Y_divide,XY_divide
        # , XY2, X2, Y2
        DisX[k] = dummyX
        dummyY = np.array([X, Y, YX2, X2, Y2, XY])  # Y3, YX4, X2Y3, Y5,   ,X_divide,Y_divide,XY_divide
        # , YX2, X2, Y2
        DisY[k] = dummyY

    # regX = LinearSVR()
    # regX = SVR()
    regX = linear_model.LinearRegression()
    regX.fit(DisX, RealWorldMatX)
    # regY = LinearSVR()
    # regY = SVR()
    regY = linear_model.LinearRegression()
    regY.fit(DisY, RealWorldMatY)
    print("X_SCORE =" + str(regX.score(DisX, RealWorldMatX)))
    print("Y_SCORE =" + str(regY.score(DisY, RealWorldMatY)))
    joblib.dump(regX, 'X_' + camera_name + '.gz', True)
    joblib.dump(regY, 'Y_' + camera_name + '.gz', True)


def m_click(event, x, y, k, l):
    if (event == cv2.EVENT_LBUTTONUP):
        print(x, y)


# (96,0),(355,15),(81,476),(623,429)
if FROM_FILE:
    pass

# cam = Retinutella(CAM_BOTTOM_MIDDLE_NAME, CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION,
                  # CAM_BOTTOM_MIDDLE_MODE, CAM_BOTTOM_MIDDLE_FOUR_POINTS, CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                  # CAM_BOTTOM_MIDDLE_MINIMUM_AREA, CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                  # CAM_BOTTOM_MIDDLE_LENGTH_PERCENT, CAM_BOTTOM_MIDDLE_BOUNDARY,
                  # CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)
cam = Retinutella(CAM_NAME, 1, CAM_ORIENT, cameraMode=1, four_points=CAM_FOUR_POINT)
# cam.getImage()
# cam = cv2.VideoCapture(1)
# ret,im = cam.read()
if FROM_FILE:
    im = cv2.imread(LOAD_IMAGE_NAME, cv2.IMREAD_COLOR)
# else:
#     ret,im = cam.read()
# cv2.setMouseCallback('image1',m_click)
while (1):
    # ret,im = cam.read()
    if not FROM_FILE:
        capture, im, matri = cam.getImage(remove_pers=True)
    elif FROM_FILE and CLASS:
        capture, im, matri = cam.getImage(remove_pers=True, LOAD_IMAGE=True, FILE_NAME_or_PATH=LOAD_IMAGE_NAME)
    else:
        im = cv2.imread(LOAD_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
    # ret, im = cam.read()
    cv2.imshow('image1', im)
    cv2.setMouseCallback('image1', m_click)
    # cv2.imwrite(SAVE_IMAGE_NAME, capture)
    k = cv2.waitKey(100)
    if k == ord('r'):
        if FROM_FILE and CLASS:
            capture, im, matri = cam.getImage(remove_pers=True, LOAD_IMAGE=True, FILE_NAME_or_PATH=LOAD_IMAGE_NAME)
        elif FROM_FILE:
            im = cv2.imread(LOAD_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
        '''shift x y
            input real world coordinate of bottom left square
        '''
        Regression_HaHA(im, camera_name=SAVED_MODEL_NAME, number_of_points=NUMBER_OF_POINTS, kernel_size=KERNEL_SIZE,
                        binarization_thresh_kernel_size=19, difference_distance_per_point=DIFFERENCE_DISTANCE_PER_POINT,
                        shift_x=SHIFT_X,
                        shift_y=SHIFT_Y)
    elif k == ord('e'):
        break
    elif k == ord('s'):
        cv2.imwrite(SAVE_IMAGE_NAME, capture)
    else:
        pass
if TEST_MODEL:
    while 1:
        #
        if FROM_FILE and CLASS:
            capture, im, matri = cam.getImage(remove_pers=True, LOAD_IMAGE=True, FILE_NAME_or_PATH=LOAD_IMAGE_NAME)
        elif FROM_FILE:
            im = cv2.imread(LOAD_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
        else:
            capture, im, matri = cam.getImage(remove_pers=True)
        cv2.imshow('image1', im)
        print('test_model')
        cv2.imshow('image1', im)
        model_X = joblib.load('X_' + MODEL_FILE_NAME + '.gz')
        model_Y = joblib.load('Y_' + MODEL_FILE_NAME + '.gz')
        cv2.waitKey(0)
        j = int(input('X value'))
        k = int(input('Y value'))
        print(type(j), type(k))
        feature_x, feature_y = IP.get_XY_feature([j, k])
        print(model_X.predict([feature_x]))
        print(model_Y.predict([feature_y]))
