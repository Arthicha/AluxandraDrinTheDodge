#import library
import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
# Measures are in milimeters

def Regression_HaHA(Image_naja,kernel_size=(4,4),binarization_thresh_kernel_size = 21,number_of_points = (12,17),difference_distance_per_point = [-120,-120],shift_x=2040,shift_y=1320,camera_name='left'):
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
    imgBW = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    img = IP.binarize(imgBW,IP.SAUVOLA_THRESHOLDING,binarization_thresh_kernel_size)
    # ret, img = cv2.threshold(imgBW, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('gray',img)
    cv2.waitKey(0)
    img = cv2.bitwise_not(img)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw contours and find centers
    for i in range(0, len(contours)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cnt = contours[i]
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        conWorldMatX = np.append(conWorldMatX, [cx])
        conWorldMatY = np.append(conWorldMatY, [cy])
        mc = (cx, cy)
        cv2.circle(imgC, mc, 3, color, -1, 8, 0)
    conWorldMatX = np.reshape(conWorldMatX,(-1,number_of_points[1]))
    conWorldMatX = np.sort(conWorldMatX,axis=1)
    conWorldMatX = np.reshape(conWorldMatX,-1)
    print('-------------')
    print('center X:')
    print(conWorldMatX)
    print('center Y:')
    print(conWorldMatY)
    print('------------------')
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
    for l in range(0,number_of_points[0]):
        RealWorldRolY[:1] = (shift_y) + (difference_distance_per_point[1]* l)
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
    DisX = np.zeros(shape=(number_of_points[0]*number_of_points[1], 8))
    DisY = np.zeros(shape=(number_of_points[0]*number_of_points[1], 8))
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
        dummyX = np.array([X3, XY2, X5, X3Y2, XY4, X2, Y2, XY])
        DisX[k] = dummyX
        dummyY = np.array([YX2, Y3, YX4, X2Y3, Y5, X2, Y2, XY])
        DisY[k] = dummyY

    regX = linear_model.LinearRegression()
    regX.fit(DisX, RealWorldMatX)
    regY = linear_model.LinearRegression()
    regY.fit(DisY, RealWorldMatY)
    print("X_SCORE =" + str(regX.score(DisX, RealWorldMatX)))
    print("Y_SCORE =" + str(regY.score(DisY, RealWorldMatY)))
    joblib.dump(regX,'X_'+camera_name+'.gz',True)
    joblib.dump(regY, 'Y_' + camera_name+'.gz', True)

def m_click (event,x,y,k,l):
    if (event == cv2.EVENT_LBUTTONUP):
        print(x,y)



cam = cv2.VideoCapture(2)
# ret,im = cam.read()
im =cv2.imread('testimBr.jpg')
cv2.imshow('image1',im)
cv2.setMouseCallback('image1',m_click)
while(1):
    # ret,im = cam.read()
    im =cv2.imread('testimBr.jpg')
    cv2.imshow('image1',im)
    cv2.setMouseCallback('image1', m_click)
    k=cv2.waitKey(100)
    if k == ord('r'):
        im = cv2.imread('testim.jpg',cv2.IMREAD_COLOR)
        Regression_HaHA(im,number_of_points=(9,19),kernel_size=[5,5],binarization_thresh_kernel_size=15,difference_distance_per_point=[-150,-150],shift_x=2900,shift_y=1275)
    elif k== ord('e'):
        break
    elif k== ord('s'):
        cv2.imwrite('testimBr.jpg',im)
    else:
        pass