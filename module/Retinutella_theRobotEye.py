__author__ = ['Zumo Arthicha Srisuchinnawong']
__version__ = 2.0
__description__ = 'Class of camera'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import copy
import os

import numpy as np
import cv2

from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP

'''*************************************************
*                                                  *
*               class Retinutella                  *
*                                                  *
*************************************************'''

class Retinutella():



    name = None             # camera name
    cam = None              # object of camera
    cameraPort = 0          # camera usb port
    cameraMode = 1          # camera mode 0 for BGR and 1 for Grayscale
    cameraOreintation = 0   # angle of rotation of camera (degree)

    windows = []


    # camera mode
    ROD = 1
    CONE = 0


    def __init__(self,name,cameraPort,cameraOreintation,cameraMode=1):

        '''
        :param name: camera name
        :param cameraPort: camera usb port
        :param cameraOreintation: angle of rotation of camera (degree)
        :param cameraMode: camera mode 0 for BGR and 1 for Grayscale
        :return: None
        example
                CAM1 = Retinutella('front',0,90,cameraMode=1)
            this line of code create an object CAM1 for camera named 'front'.
        It connect to the camera usb port 0. The image capturing from this camera
        is rotated 90 degree and converted to gray scale.
        '''

        self.name = name
        self.cameraPort = cameraPort
        self.cameraMode = cameraMode
        self.cameraOreintation = cameraOreintation
        self.cam = cv2.VideoCapture(self.cameraPort)


    def getImage(self,fileName=None):

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


        ret, img = self.cam.read(self.cameraMode)
        if self.cameraMode == self.ROD:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.cameraMode is self.CONE:
            rows,cols,_ = img.shape
        else:
            rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.cameraOreintation,1)
        img = cv2.warpAffine(img,M,(cols,rows))
        path = os.getcwd()
        if fileName != None:
            cv2.imwrite(path+fileName,img)
        return img

    def getListOfPlate(self):
        image = self.getImage()
        #ret, image = cv2.threshold(image, 100, 255,0)
        plate = IP.Get_Plate2(image)
        plate = IP.Get_Word2(plate)
        #listOfImage = IP.get_plate(image,(64, 32))
        #print('return from get plate',listOfImage)
        return image,plate


    def close(self):

        '''
            CAM1.close()
            destroy object of camera.
        '''

        del self.cam
        self.cam = None




    def show(self,image,frame=None,wait=None):

        '''
        :param image: image to show.
        :param frame: frame name.If frame name is None, the default frame name is same as
        camera name.
        :param wait: wait time to show image. If wait is None, that means no wait time, while
        wait time is 0 mean untill keyboard interrupt.
        :return: None
        example
                                CAM1.show(image,frame='original',wait=1)
            this function create frame named 'original', it show image and wait (delay) 1 second.
        '''

        if frame == None:
            frame = self.name

        cv2.imshow(frame,image)
        if frame != self.name:
            self.windows.append(frame)
        if wait != None:
            cv2.waitKey(wait)


    def destroyWindows(self):

        '''
        destroy all windows that are shown from this object
        '''

        for window in self.windows:
            cv2.destroyWindow(window)


    '''*************************************************
    *                                                  *
    *                  private method                  *
    *                                                  *
    *************************************************'''
    def checkOreantation(self,img):

        LMG = []
        for name in ['5','twoTH','ThreeEN']:
            sample = cv2.imread(name+'.jpg')
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            ret, sample = cv2.threshold(sample, 127, 255,0)
            sample, contours, hierarchy = cv2.findContours(sample, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            sample = []
            for cnt in contours:
                sample += cnt.tolist()
            sample = np.array(sample)
            LMG.append(sample)
        img, contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        img = []
        for cnt in contours:
            img += cnt.tolist()
        img = np.array(img)

        p_ret = cv2.matchShapes(img,LMG[0],1,0.0)
        for i in range(1,len(LMG)):
            ret = cv2.matchShapes(img,LMG[i],1,0.0)
            if ret < p_ret:
                p_ret = ret
        return ret


    def aspectRatio(self,img_f):
        img_fc = copy.deepcopy(img_f)
        img_fc, cfc, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            xfc,yfc,wfc,hfc = cv2.boundingRect(cfc[-1])
        except:
            return 1.0
        aspect_ratio = float(wfc)/hfc
        return aspect_ratio

    def getWordSize(self,img_f):
        img_fc = copy.deepcopy(img_f)
        img_fc, contours, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            cnt = contours[-1]
        except:
            return 0,0
        leftmost = np.array(cnt[cnt[:,:,0].argmin()][0])
        rightmost = np.array(cnt[cnt[:,:,0].argmax()][0])
        topmost = np.array(cnt[cnt[:,:,1].argmin()][0])
        bottommost = np.array(cnt[cnt[:,:,1].argmax()][0])
        return np.linalg.norm(leftmost-rightmost),np.linalg.norm(topmost-bottommost)




