

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
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points=((0,0),(300,300),(0,300),(300,0)),thresh_kernel = 21, minimum_area=0.01,maximum_area = 0.9,lengthpercent =0.01,thresh_kernel_original = 21, minimum_area_original=0.01,maximum_area_original = 0.9,lengthpercent_original =0.01,word_boundary=20,binarize_method=IP.SAUVOLA_THRESHOLDING,Offset_homo_x = 0,Offset_homo_y=0,closing_kernel_size = 5):
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
        self.minimum_area= minimum_area
        self.maximum_area = maximum_area
        self.length_percent= lengthpercent
        self.thresh_kernel= thresh_kernel

        self.minimum_area_original = minimum_area_original
        self.maximum_area_original = maximum_area_original
        self.length_percent_original = lengthpercent_original
        self.thresh_kernel_original = thresh_kernel_original
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size

    def getListOfPlate(self,image_size=(30,60),platePos=False,plateOrientation = False,show=False,LOAD_IMAGE =False,FILENAME='picture\\testpic\TestBottomRightSide.jpg'):
        ''' '''
        '''ConFig parameter of bottom_middle left and right side'''

        # MINIMUM_AREA_ORIGINAL_PIC = 0.01
        # LENGPERCENT_ORIGINAL_PIC = 0.01
        # MINIMUM_AREA_OTHER_PIC = 0.01
        # LENGPERCENT_OTHER_PIC = 0.01
        ''' ****************************************    '''

        '''have been edited'''
        image,capture,matrice= self.getImage(remove_pers=True,LOAD_IMAGE =LOAD_IMAGE,FILE_NAME_or_PATH=FILENAME)
        if platePos and plateOrientation:
            plate_capture, plate_pos,plate_orientation = IP.Get_Plate2(capture,thres_kirnel=self.thresh_kernel,max_area= self.maximum_area,min_area=self.minimum_area,lengPercent=self.length_percent, center=True, before=True,orientation=True,model_x=self.model_x,model_y=self.model_y,binarization_method=self.binarize_method,closing_kernel_size=self.closing_kernel_size)
        elif platePos:
            ''' have been edited'''
            plate_capture,plate_pos = IP.Get_Plate2(capture,thres_kirnel=self.thresh_kernel,max_area= self.maximum_area,min_area=self.minimum_area,lengPercent=self.length_percent,center=True,before=True,binarization_method=self.binarize_method)
            '''end of edit'''

        #ret, image = cv2.threshold(image, 100, 255,0)
        plate,platePos_ = IP.Get_Plate2(image,thres_kirnel=self.thresh_kernel_original,max_area= self.maximum_area_original,min_area=self.minimum_area_original,lengPercent=self.length_percent_original,center=True,before=True,binarization_method=self.binarize_method,closing_kernel_size=self.closing_kernel_size)
        # print(plate)
        plate = IP.Get_Word2(plate,image_size=image_size,boundary=self.boundary)
        # print(plate)
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
            new_position=list(new_position[0][0:2])
            return new_position

        def regress_to_real_world(self,points):
            if points!= ():
                feature_x,feature_y = IP.get_XY_feature(points)
                new_x = self.model_x.predict([feature_x])
                new_y = self.model_y.predict([feature_y])
                if 'L' in self.name:
                    return [-500 + self.z, new_x[0], new_y[0][0]]
                elif 'R' in self.name:
                    return [500 - self.z, new_x[0], new_y[0][0]]
                elif 'B' in self.name:
                    return [new_x[0],new_y[0][0],self.z]
            else:
                return ()

        def orientation_to_mat(self,orientation):
            if orientation is None:
                return None
            else:
                if 'L' in self.name:
                    return np.array([[0, 0, -1],[np.cos(orientation[0]), np.sin(orientation[0]), 0],
                              [np.sin(orientation[0]), -np.cos(orientation[0]), 0]])
                elif 'R' in self.name:
                    return np.array([[0, 0, 1], [np.cos(orientation[0]), np.sin(orientation[0]), 0],
                              [-np.sin(orientation[0]), np.cos(orientation[0]), 0]])
                elif 'B' in self.name:
                    return np.array([[np.cos(orientation[0]), np.sin(orientation[0]), 0], [np.sin(orientation[0]), -np.cos(orientation[0]), 0],
                              [0,0,-1]])

        print(platePos_)
        # platePos_ = list(map(lambda x:calculate_position(x,matrice),platePos_))
        platePos_ = list(map(lambda x:[x[0]+self.Offset_homo_x,x[1]+self.Offset_homo_y],platePos_))
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
                # print(plate_pos)
                # print(plate_orientation)
                # print("**********")
                # print(index)
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

        plate, sorted_plate_pos, sorted_plate_orientation = IP.filter_plate(plate, sorted_plate_pos,  sorted_plate_orientation)
        '''end'''
        print(len(plate),platePos,plateOrientation)
        if platePos and plateOrientation:
            return image, plate, sorted_plate_pos,sorted_plate_orientation
        elif platePos:
            ''' end of my part'''
            return image,plate,sorted_plate_pos
        else:
            return image,plate

class Camera_right(Camera_left):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points=((0,0),(300,300),(0,300),(300,0)),thresh_kernel = 21, minimum_area=0.01,maximum_area = 0.9,lengthpercent =0.01,thresh_kernel_original = 21, minimum_area_original=0.01,maximum_area_original = 0.9,lengthpercent_original =0.01,word_boundary=20,binarize_method=IP.SAUVOLA_THRESHOLDING,Offset_homo_x = 0,Offset_homo_y=0,closing_kernel_size=5):
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
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.length_percent = lengthpercent
        self.thresh_kernel = thresh_kernel
        # self.minimum_area = minimum_area
        # self.maximum_area = maximum_area
        # self.length_percent = lengthpercent
        # self.thresh_kernel = thresh_kernel
        self.minimum_area_original = minimum_area_original
        self.maximum_area_original = maximum_area_original
        self.length_percent_original = lengthpercent_original
        self.thresh_kernel_original = thresh_kernel_original
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size


class Camera_Bottom_middle(Camera_left):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points=((0,0),(300,300),(0,300),(300,0)),thresh_kernel = 21, minimum_area=0.01,maximum_area = 0.9,lengthpercent =0.01,thresh_kernel_original = 21, minimum_area_original=0.01,maximum_area_original = 0.9,lengthpercent_original =0.01,word_boundary=20,binarize_method=IP.SAUVOLA_THRESHOLDING,Offset_homo_x = 0,Offset_homo_y=0,closing_kernel_size=5):
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
        self.name ='Bm'
        self.model_x = joblib.load(model_path+'X_'+self.name+'.gz')
        self.model_y = joblib.load(model_path+'Y_'+self.name+'.gz')
        self.z = offset_z
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.length_percent = lengthpercent
        self.thresh_kernel = thresh_kernel
        # self.minimum_area = minimum_area
        # self.maximum_area = maximum_area
        # self.length_percent = lengthpercent
        # self.thresh_kernel = thresh_kernel
        self.minimum_area_original = minimum_area_original
        self.maximum_area_original = maximum_area_original
        self.length_percent_original = lengthpercent_original
        self.thresh_kernel_original = thresh_kernel_original
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size

class Camera_Bottom_right(Camera_left):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points_bottom=[[0,0],[300,300],[0,300],[300,0]],four_points_side = [[0,0],[300,300],[0,300],[300,0]],thresh_kernel = 21, minimum_area=0.01,maximum_area = 0.9,lengthpercent =0.15,thresh_kernel_original = 21, minimum_area_original=0.01,maximum_area_original = 0.9,lengthpercent_original =0.01,word_boundary=20,binarize_method=IP.SAUVOLA_THRESHOLDING,Offset_homo_x = 0,Offset_homo_y=0,closing_kernel_size=5):
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
        super().__init__(cameraPort,cameraOreintation,cameraMode=cameraMode,four_points=four_points_bottom)
        self.four_points_side = four_points_side
        self.four_points_bottom = four_points_bottom
        self.name ='Br'
        self.model_x_side = joblib.load(model_path+'X_'+self.name+'_side.gz')
        self.model_y_side = joblib.load(model_path+'Y_'+self.name+'_side.gz')
        self.model_x_bottom = joblib.load(model_path + 'X_' + self.name + '_bottom.gz')
        self.model_y_bottom = joblib.load(model_path + 'Y_' + self.name + '_bottom.gz')
        self.z = offset_z
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.length_percent = lengthpercent
        self.thresh_kernel = thresh_kernel
        # self.minimum_area = minimum_area
        # self.maximum_area = maximum_area
        # self.length_percent = lengthpercent
        # self.thresh_kernel = thresh_kernel
        self.minimum_area_original = minimum_area_original
        self.maximum_area_original = maximum_area_original
        self.length_percent_original = lengthpercent_original
        self.thresh_kernel_original = thresh_kernel_original
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size

    def getImage(self, fileName=None, remove_pers=False, show=True,LOAD_IMAGE=False,FILE_NAME_or_PATH = 'picture\\testpic\TestBottomRightSide.jpg'):

        '''
        :param fileName: file to save an image which captured from this object, this
        parameter will be set to None if you want only to get captured image.
        :return: an image captured from this camera object.
        example
                image = CAM1.getImage(fileName='imageCapture1.jpg')
            this function will return image captured from CAM1 object with configuration
        that had been set in object constructor. In addition, the program will save that
        image as 'imageCapture1.jpg'.
        '''
        if LOAD_IMAGE:
            img = cv2.imread(FILE_NAME_or_PATH,cv2.IMREAD_GRAYSCALE)
        else:
            ret, img = self.cam.read(self.cameraMode)

        if self.cameraMode == self.ROD:
            if LOAD_IMAGE:
                pass
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.cameraMode is self.CONE:
            rows, cols, _ = img.shape
        else:
            rows, cols = img.shape
        maxima = max(img.shape)
        # print(maxima)
        blank_image = np.ones((cols, cols), np.uint8) * 255
        blank_image[int((maxima - rows) / 2):int(maxima - (maxima - rows) / 2),
        int((maxima - cols) / 2):int(maxima - (maxima - cols) / 2)] = img
        new_image_rows, new_image_cols = blank_image.shape
        # blank_image[:,:]
        M = cv2.getRotationMatrix2D((maxima / 2, maxima / 2), self.cameraOreintation, 1)
        img = cv2.warpAffine(blank_image, M, (cols, cols), borderValue=255)
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.cameraOreintation, 1)
        # img = cv2.warpAffine(img, M, (cols, rows))
        if show is True:
            def m_click(event, x, y, k, l):
                if (event == cv2.EVENT_LBUTTONUP):
                    print(x, y)

            cv2.imshow('image', img)
            cv2.setMouseCallback('image', m_click)
            cv2.waitKey(100)

        path = os.getcwd()
        if fileName != None:
            cv2.imwrite(path + fileName, img)
        ''' my code '''
        if remove_pers:
            capture, matrice = IP.four_point_transform(img, self.four_points_bottom, True)
            capture2, matrice2 = IP.four_point_transform(img, self.four_points_side, True)
            return img, capture, matrice,capture2, matrice2
        '''end of my code'''
        return img


    def getListOfPlate(self, image_size=(30, 60), platePos=False, plateOrientation=False, show=False,LOAD_IMAGE =False,FILENAME='picture\\testpic\TestBottomRightSide.jpg' ):
        '''have been edited'''
        '''ConFig parameter of bottom_left and bottom_right side'''

        MINIMUM_AREA_ORIGINAL_PIC = 0.01
        LENGPERCENT_ORIGINAL_PIC = 0.01
        MINIMUM_AREA_OTHER_PIC = 0.01
        LENGPERCENT_OTHER_PIC = 0.15
        ''' ****************************************    '''

        image, capture, matrice , capture2 , matrice2 = self.getImage(remove_pers=True,LOAD_IMAGE=LOAD_IMAGE,FILE_NAME_or_PATH= FILENAME)

        if platePos and plateOrientation:
            plate_capture, plate_pos, plate_orientation = IP.Get_Plate2(capture,thres_kirnel=self.thresh_kernel,max_area= self.maximum_area,min_area=self.minimum_area,lengPercent=self.length_percent, center=True,
                                                                        before=True, orientation=True,model_x=self.model_x_bottom,model_y=self.model_y_bottom,closing_kernel_size=self.closing_kernel_size)
            plate_capture2, plate_pos2, plate_orientation2 = IP.Get_Plate2(capture2,thres_kirnel=21,max_area= self.maximum_area,min_area=self.minimum_area,lengPercent=self.length_percent, center=True,
                                                                        before=True, orientation=True,model_x=self.model_x_side,model_y=self.model_y_side,closing_kernel_size=self.closing_kernel_size)
        elif platePos:
            ''' have been edited'''
            plate_capture, plate_pos = IP.Get_Plate2(capture,thres_kirnel=self.thresh_kernel,max_area= self.maximum_area,min_area=self.minimum_area,lengPercent=self.length_percent, center=True,
                                                                        before=True, orientation=True,model_x=self.model_x_side,model_y=self.model_y_side)
            '''end of edit'''

        # ret, image = cv2.threshold(image, 100, 255,0)
        plate, platePos_ = IP.Get_Plate2(image,thres_kirnel=self.thresh_kernel_original,max_area= self.maximum_area_original,min_area=self.minimum_area_original,lengPercent=self.length_percent_original, center=True, before=True,closing_kernel_size=self.closing_kernel_size)
        plate = IP.Get_Word2(plate, image_size=image_size,boundary=self.boundary)
        print('............')
        print(len(plate))
        print('............')
        # listOfImage = IP.get_plate(image,(64, 32))
        # print('return from get plate',listOfImage)
        ''' my part '''
        # print(plate)
        if show:
            show_capture = copy.deepcopy(capture)
            show_capture = cv2.cvtColor(show_capture, cv2.COLOR_GRAY2BGR)
            show_capture2 = copy.deepcopy(capture2)
            show_capture2 = cv2.cvtColor(show_capture2, cv2.COLOR_GRAY2BGR)
            for i in plate_pos:
                pos = (int(i[0]), int(i[1]))
                cv2.circle(show_capture, pos, 3, [255, 0, 0])
            cv2.imshow("capture", show_capture)
            cv2.imshow("capture2", show_capture2)
            cv2.waitKey(100)

        def calculate_position(position, HomoMatrix):
            ''' for 2 dimen only'''
            new_position = (position[0], position[1], 1)
            new_position = np.matmul(HomoMatrix, np.reshape(new_position, (-1, 1)))
            new_position = np.reshape(new_position, (1, -1)).tolist()
            new_position = list(new_position[0][0:2])
            return new_position

        def regress_to_real_world( points,model_x,model_y,side=False):
            if side is False:
                if points != ():
                    feature_x, feature_y = IP.get_XY_feature(points)
                    new_x = model_x.predict([feature_x])
                    new_y = model_y.predict([feature_y])
                    if 'L' in self.name:
                        return [-500 + self.z, new_x[0], new_y[0][0]]
                    elif 'R' in self.name:
                        return [500-self.z, new_x[0], new_y[0][0]]
                    elif 'B' in self.name:
                        return [new_x[0],new_y[0][0],self.z]
                else:
                    return ()
            else:
                if points != ():
                    feature_x, feature_y = IP.get_XY_feature(points)
                    new_x = model_x.predict([feature_x])
                    new_y = model_y.predict([feature_y])
                    if 'l' in self.name:
                        return [-500 + self.z, new_y[0][0],new_x[0] ]
                    if 'r' in self.name:
                        return [500-self.z,new_y[0][0] , new_x[0]]
                else:
                    return ()

        def orientation_to_mat(self, orientation,side=False):
            if side is False:
                if orientation is None:
                    return None
                else:
                    if 'L' in self.name:
                        return np.array([[0, 0, -1],
                                         [np.cos(orientation[0]), np.sin(orientation[0]), 0],
                                        [np.sin(orientation[0]), -np.cos(orientation[0]), 0]])
                    elif 'R' in self.name:
                        return np.array([[0, 0, 1],
                                         [np.cos(orientation[0]), np.sin(orientation[0]), 0],
                                        [-np.sin(orientation[0]), np.cos(orientation[0]), 0]])
                    elif 'B' in self.name:
                        return np.array([[np.cos(orientation[0]), np.sin(orientation[0]), 0],
                                         [np.sin(orientation[0]), -np.cos(orientation[0]), 0],
                                        [0,0,-1]])
            else:
                if orientation is None:
                    return None
                else:
                    if 'l' in self.name:
                        return np.array([[0, 0, -1],[np.cos(orientation[0]), np.sin(orientation[0]), 0],
                                  [np.sin(orientation[0]), -np.cos(orientation[0]), 0]])
                    elif 'r' in self.name:
                        return np.array([[0, 0, 1], [np.cos(orientation[0]), np.sin(orientation[0]), 0],
                                  [-np.sin(orientation[0]), np.cos(orientation[0]), 0]])

        # platePos_dum = list(map(lambda x: calculate_position(x, matrice), platePos_))
        # print(platePos_dum)
        # print(platePos_)
        platePos_dum = list(map(lambda x: [x[0] + self.Offset_homo_x, x[1] + self.Offset_homo_y], platePos_))
        sorted_plate_pos = [x for x in platePos_dum]
        sorted_plate_orientation = [x for x in platePos_dum]
        # print(platePos_)
        ''' sorted plate pos in here'''

        if platePos:
            if platePos_dum != [] and plate_pos != []:
                tree = cKDTree(platePos_dum)
                dist, index = tree.query(plate_pos)
                print("***********")
                print(index)
                print(platePos_dum)
                print(plate_pos)
                print(plate_orientation)
                print("**********")
                if platePos and plateOrientation:
                    for x, y, z in zip(index, plate_pos, plate_orientation):
                        sorted_plate_pos[x] = y
                        sorted_plate_orientation[x] = z

                else:
                    for x, y in zip(index, plate_pos):
                        sorted_plate_pos[x] = y

                for x in range(0, len(sorted_plate_pos)):
                    if x in index:
                        pass
                    else:
                        sorted_plate_pos[x] = ()
                        sorted_plate_orientation[x] = None
                        # print(sorted_plate_pos)
                        # print("----------------")
        print(sorted_plate_pos)
        # print(sorted_plate_pos)
        print("----------------")
        print(len(plate))
        sorted_plate_pos = list(map(lambda x: regress_to_real_world( x,self.model_x_bottom,self.model_y_bottom), sorted_plate_pos))
        sorted_plate_orientation = list(map(lambda x: orientation_to_mat(self, x), sorted_plate_orientation))

        platePos_dum = list(map(lambda x: calculate_position(x, matrice2), platePos_))
        sorted_plate_pos2 = [x for x in platePos_dum]
        sorted_plate_orientation2 = [x for x in platePos_dum]
        # print(platePos_)
        ''' sorted plate pos in here'''

        # if platePos:
        #     if platePos_dum != [] and plate_pos2 != []:
        #         tree = cKDTree(platePos_dum)
        #         dist, index = tree.query(plate_pos2)
        #         # print("***********")
        #         # # print(index)
        #         # # print(platePos_)
        #         # print(plate_pos2)
        #         # print(plate_orientation2)
        #         # print("**********")
        #         if platePos and plateOrientation:
        #             for x, y, z in zip(index, plate_pos2, plate_orientation2):
        #                 sorted_plate_pos2[x] = y
        #                 sorted_plate_orientation2[x] = z
        #
        #         else:
        #             for x, y in zip(index, plate_pos2):
        #                 sorted_plate_pos2[x] = y
        #
        #         for x in range(0, len(sorted_plate_pos2)):
        #             if x in index:
        #                 pass
        #             else:
        #                 sorted_plate_pos2[x] = ()
        #                 sorted_plate_orientation2[x] = None
        #                 # print(sorted_plate_pos)
        #                 # print("----------------")
        #         sorted_plate_pos2 = list(
        #             map(lambda x: regress_to_real_world(x, self.model_x_side, self.model_y_side,side=True), sorted_plate_pos2))
        #         sorted_plate_orientation2 = list(map(lambda x: orientation_to_mat(self, x,side=True), sorted_plate_orientation2))
        #
        #         for i in range(0,len(sorted_plate_pos2)):
        #             if sorted_plate_pos[i] == ():
        #                 sorted_plate_pos[i] = sorted_plate_pos2[i]
        #                 sorted_plate_orientation[i] = sorted_plate_orientation2[i]
        # print(plate)
        plate, sorted_plate_pos, sorted_plate_orientation =IP.filter_plate(plate, sorted_plate_pos, sorted_plate_orientation)
        '''end'''
        # plate, sorted_plate_pos, sorted_plate_orientation = IP.find_same_point_and_average_it(plate,
        #                                                                                           sorted_plate_pos,
        #                                                                                           sorted_plate_orientation,                                                                  75)
        if platePos and plateOrientation:
            return image, plate, sorted_plate_pos, sorted_plate_orientation
        elif platePos:
            ''' end of my part'''
            return image, plate, sorted_plate_pos
        else:
            return image, plate


class Camera_Bottom_left(Camera_Bottom_right):
    def __init__(self,cameraPort,cameraOreintation,cameraMode=1,offset_z = 50,four_points_bottom=np.array([[0,0],[300,300],[0,300],[300,0]]),four_points_side = np.array([[300,0],[600,300],[300,300],[600,0]]),thresh_kernel = 21, minimum_area=0.01,maximum_area = 0.9,lengthpercent =0.15,thresh_kernel_original = 21, minimum_area_original=0.01,maximum_area_original = 0.9,lengthpercent_original =0.01,word_boundary=20,binarize_method=IP.SAUVOLA_THRESHOLDING,Offset_homo_x = 100,Offset_homo_y=100,closing_kernel_size=5):
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
        super().__init__(cameraPort,cameraOreintation,cameraMode=cameraMode,four_points_bottom=four_points_bottom,four_points_side=four_points_side)
        self.name = 'Bl'
        self.model_y_side = joblib.load(model_path + 'X_' + self.name + '_side.gz')
        self.model_y_side = joblib.load(model_path + 'Y_' + self.name + '_side.gz')
        self.model_x_bottom = joblib.load(model_path + 'X_' + self.name + '_bottom.gz')
        self.model_y_bottom = joblib.load(model_path + 'Y_' + self.name + '_bottom.gz')
        self.z = offset_z
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.length_percent = lengthpercent
        self.thresh_kernel = thresh_kernel
        # self.minimum_area = minimum_area
        # self.maximum_area = maximum_area
        # self.length_percent = lengthpercent
        # self.thresh_kernel = thresh_kernel
        self.minimum_area_original = minimum_area_original
        self.maximum_area_original = maximum_area_original
        self.length_percent_original = lengthpercent_original
        self.thresh_kernel_original = thresh_kernel_original
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size





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
