

import copy
import os
import random
import numpy as np
import cv2
from scipy.spatial import cKDTree
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib

class Camera_left(Retinutella):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points=((0,0),(300,300),(0,300),(300,0))):

        '''
        :param name: camera name
        :param cameraPort: camera usb port
        :param cameraOreintation: angle of rotation of camera (degree)
        :param cameraMode: camera mode 0 for BGR and 1 for Grayscale
        :param four_points: four point to extract image
        :param offset_z: distance of z against a wall
        :return: None
        example
                CAM1 =  Camera_left(0,90,cameraMode=1)
            this line of code create an object  Camera_left name 'L'
        It connect to the camera usb port 0. The image capturing from this camera
        is rotated 90 degree and converted to gray scale.
        '''
        current_path = os.getcwd()
        model_path = current_path.split(os.sep)
        model_path = model_path[:]
        model_path = os.sep.join(model_path)
        model_path = model_path+os.sep+'savedModel'+os.sep+'modelcamera'+os.sep
        super().__init__('L',cameraPort,cameraOreintation,cameraMode=cameraMode,four_points=four_points)
        self.model_x = joblib.load(model_path+'X_'+self.name+'.gz')
        self.model_y = joblib.load(model_path+'Y_'+self.name+'.gz')
        self.z = offset_z

    def getListOfPlate(self,image_size=(30,60),platePos=False,plateOrientation = False,show=False):
        '''have been edited'''
        image,capture,matrice= self.getImage(remove_pers=True)
        if platePos and plateOrientation:
            plate_capture, plate_pos,plate_orientation = IP.Get_Plate2(capture, min_area=0.01, center=True, before=True,orientation=True)
        elif platePos:
            ''' have been edited'''
            plate_capture,plate_pos = IP.Get_Plate2(capture,min_area=0.01,center=True,before=True)
            '''end of edit'''


        #ret, image = cv2.threshold(image, 100, 255,0)
        plate,platePos_ = IP.Get_Plate2(image,min_area=0.01,center=True,before=True)
        plate = IP.Get_Word2(plate,image_size=image_size)
        #listOfImage = IP.get_plate(image,(64, 32))
        #print('return from get plate',listOfImage)
        ''' my part '''


        if show:
            show_capture = copy.deepcopy(capture)
            show_capture= cv2.cvtColor(show_capture,cv2.COLOR_GRAY2BGR)
            for i in plate_pos:
                pos = (int(i[0]),int(i[1]))
                cv2.circle(show_capture,pos,3,[255,0,0])
            cv2.imshow("capture",show_capture)
            cv2.waitKey(100)

        def calculate_position(position,HomoMatrix):
            ''' for 2 dimen only'''
            new_position =(position[0],position[1],1)
            new_position = np.matmul(HomoMatrix,np.reshape(new_position,(-1,1)))
            new_position=np.reshape(new_position,(1,-1)).tolist()
            new_position=tuple(new_position[0][0:2])
            return new_position

        def regress_to_real_world(self,points):
            if points!= ():
                feature_x,feature_y = IP.get_XY_feature(points)
                new_x = self.model_x.predict([feature_x])
                new_y = self.model_y.predict([feature_y])
                if 'L' in self.name:
                    return [new_x[0],self.z,new_y[0][0]]
                elif 'R' in self.name:
                    return [new_x[0],self.z,new_y[0][0]]
                elif 'B' in self.name:
                    return [new_y[0][0],new_x[0],self.z]
            else:
                return ()

        def orientation_to_mat(self,orientation):
            if orientation is None:
                return None
            else:
                if 'L' in self.name:
                    return np.array([[np.cos(-orientation[0]),np.sin(-orientation[0]),0],[0,0,1],[np.sin(-orientation[0]),np.cos(-orientation[0]),0]])
                elif 'R' in self.name:
                    return np.array([[-np.cos(orientation[0]),-np.sin(orientation[0]),0],[0,0,-1],[np.sin(orientation[0]),-np.cos(orientation[0]),0]])
                elif 'B' in self.name:
                    return np.array([[-np.cos(orientation[0]),-np.sin(orientation[0]),0],[0,0,-1],[np.sin(orientation[0]),-np.cos(orientation[0]),0]])

        platePos_ = list(map(lambda x:calculate_position(x,matrice),platePos_))
        sorted_plate_pos = [x for x in platePos_]
        sorted_plate_orientation = [x for x in platePos_]
        # print(platePos_)
        ''' sorted plate pos in here'''

        if platePos:
            if platePos_ != [] and plate_pos != []:
                tree = cKDTree(platePos_)
                dist, index = tree.query(plate_pos)
                print("***********")
                # print(index)
                # print(platePos_)
                print(plate_pos)
                print(plate_orientation)
                print("**********")
                if platePos and plateOrientation:
                    for x,y,z in zip(index,plate_pos,plate_orientation):
                        sorted_plate_pos[x]=y
                        sorted_plate_orientation[x]=z

                else:
                    for x,y in zip(index,plate_pos):
                        sorted_plate_pos[x]=y

                for x in range(0,len(sorted_plate_pos)):
                    if x in index:
                        pass
                    else:
                        sorted_plate_pos[x]=()
                        sorted_plate_orientation[x]= None
                # print(sorted_plate_pos)
                # print("----------------")

        # print(sorted_plate_pos)
        # print("----------------")
        sorted_plate_pos = list(map(lambda x:regress_to_real_world(self,x) ,sorted_plate_pos))
        sorted_plate_orientation = list(map(lambda x:orientation_to_mat(self,x),sorted_plate_orientation))
        '''end'''
        if platePos and plateOrientation:
            return image, plate, sorted_plate_pos,sorted_plate_orientation
        elif platePos:
            ''' end of my part'''
            return image,plate,sorted_plate_pos
        else:
            return image,plate

class Camera_right(Camera_left):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points=((0,0),(300,300),(0,300),(300,0))):
        '''
        :param name: camera name
        :param cameraPort: camera usb port
        :param cameraOreintation: angle of rotation of camera (degree)
        :param cameraMode: camera mode 0 for BGR and 1 for Grayscale
        :param four_points: four point to extract image
        :param offset_z: distance of z against a wall
        :return: None
        example
                CAM1 =  Camera_left(0,90,cameraMode=1)
            this line of code create an object  Camera_left name 'L'
        It connect to the camera usb port 0. The image capturing from this camera
        is rotated 90 degree and converted to gray scale.
        '''
        current_path = os.getcwd()
        model_path = current_path.split(os.sep)
        model_path = model_path[:]
        model_path = os.sep.join(model_path)
        model_path = model_path+os.sep+'savedModel'+os.sep+'modelcamera'+os.sep
        super().__init__(cameraPort,cameraOreintation,cameraMode=cameraMode,four_points=four_points)
        self.name ='R'
        self.model_x = joblib.load(model_path+'X_'+self.name+'.gz')
        self.model_y = joblib.load(model_path+'Y_'+self.name+'.gz')
        self.z = offset_z







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
