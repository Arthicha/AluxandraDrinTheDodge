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
from scipy.spatial import cKDTree
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from sklearn.externals import  joblib
'''*************************************************
*                                                  *
*               class Retinutella                  *
*                                                  *
*************************************************'''
OFFSET_ORIENTATION = 0.261799

class Retinutella():
    name = None  # camera name
    cam = None  # object of camera
    cameraPort = 0  # camera usb port
    cameraMode = 1  # camera mode 0 for BGR and 1 for Grayscale
    cameraOreintation = 0  # angle of rotation of camera (degree)

    windows = []

    # camera mode
    ROD = 1
    CONE = 0

    def __init__(self, name, cameraPort, cameraOreintation, cameraMode=1,
                 four_points=((0, 0), (300, 300), (0, 300), (300, 0)), thresh_kernel=21, minimum_area=0.01,
                 maximum_area=0.9, lengthpercent=0.01, word_boundary=20, binarize_method=IP.SAUVOLA_THRESHOLDING,
                 Offset_homo_x=0, Offset_homo_y=0, closing_kernel_size=5, offset_z=25):

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
        current_path = os.getcwd()
        model_path = current_path.split(os.sep)
        model_path = model_path[:]
        model_path = os.sep.join(model_path)
        model_path = model_path + os.sep + 'savedModel' + os.sep + 'modelcamera' + os.sep
        self.name = name
        self.cameraPort = cameraPort
        self.cameraMode = cameraMode
        self.cameraOreintation = cameraOreintation
        self.cam = cv2.VideoCapture(self.cameraPort)
        '''my code'''
        self.model_x = joblib.load(model_path + 'X_' + self.name + '.gz')
        self.model_y = joblib.load(model_path + 'Y_' + self.name + '.gz')
        self.four_points = np.reshape(np.array(four_points), (4, 2))
        self.z = offset_z
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.length_percent = lengthpercent
        self.thresh_kernel = thresh_kernel
        self.boundary = word_boundary
        self.binarize_method = binarize_method
        self.Offset_homo_x = Offset_homo_x
        self.Offset_homo_y = Offset_homo_y
        self.closing_kernel_size = closing_kernel_size
        '''end of my code'''

    def getImage(self, fileName=None, remove_pers=False, show=True, LOAD_IMAGE=False,
                 FILE_NAME_or_PATH='picture\\testpic\TestBottomRightSide.jpg'):

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
            img = cv2.imread(FILE_NAME_or_PATH, cv2.IMREAD_GRAYSCALE)
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
        blank_image = np.ones((cols, cols), np.uint8) * 255
        blank_image[int((maxima - rows) / 2):int(maxima - (maxima - rows) / 2),
        int((maxima - cols) / 2):int(maxima - (maxima - cols) / 2)] = img
        M = cv2.getRotationMatrix2D((maxima / 2, maxima / 2), self.cameraOreintation, 1)
        img = cv2.warpAffine(blank_image, M, (cols, cols), borderValue=255)
        if show is True:
            def m_click(event, x, y, k, l):
                if (event == cv2.EVENT_LBUTTONUP):
                    print(x, y)

            cv2.imshow('image', img)
            cv2.setMouseCallback('image', m_click)
            cv2.waitKey(10)
        path = os.getcwd()
        if fileName != None:
            cv2.imwrite(path + fileName, img)
        if remove_pers:
            capture, matrice = IP.four_point_transform(img, self.four_points, True)
            return img, capture, matrice
        return img, None, None

    def getListOfPlate(self, image_size=(30, 60), show=False,LOAD_IMAGE =False,FILENAME='picture\\testpic\TestBottomRightSide.jpg'):
        '''have been edited'''
        image, capture, matrice = self.getImage(remove_pers=True, LOAD_IMAGE=LOAD_IMAGE,
                 FILE_NAME_or_PATH=FILENAME)
        plate_capture, plate_pos, plate_orientation = IP.Get_Plate2(capture, thres_kirnel=self.thresh_kernel,
                                                                    max_area=self.maximum_area,
                                                                    min_area=self.minimum_area,
                                                                    lengPercent=self.length_percent, center=True,
                                                                    before=True, orientation=True,
                                                                    binarization_method=self.binarize_method,
                                                                    closing_kernel_size=self.closing_kernel_size,
                                                                    model_x=self.model_x, model_y=self.model_y)
        print(len(plate_capture))
        plate = IP.Get_Word2(plate_capture, image_size=image_size, boundary=self.boundary)
        ''' my part '''
        if show:
            show_capture = copy.deepcopy(capture)
            show_capture = cv2.cvtColor(show_capture, cv2.COLOR_GRAY2BGR)
            for i,j in zip(plate_pos,plate_orientation):
                pos = (int(i[0]), int(i[1]))
                cv2.circle(show_capture, pos, 3, [255, 0, 0])
            cv2.imshow("capture", show_capture)
            cv2.waitKey(100)

        sorted_plate_pos = list(map(lambda x: self.regress_to_real_world(x), plate_pos))
        sorted_plate_orientation = list(map(lambda x: self.orientation_to_mat(x), plate_orientation))
        plate, sorted_plate_pos, sorted_plate_orientation = IP.filter_plate(plate, sorted_plate_pos,
                                                                            sorted_plate_orientation)
        plate, sorted_plate_pos, sorted_plate_orientation = IP.find_same_point_and_average_it(plate, sorted_plate_pos, sorted_plate_orientation)
        return image, plate, sorted_plate_pos, sorted_plate_orientation

    def close(self):

        '''
            CAM1.close()
            destroy object of camera.
        '''

        del self.cam
        self.cam = None

    def show(self, image, frame=None, wait=None):

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

        cv2.imshow(frame, image)
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

    def checkOreantation(self, img):

        LMG = []
        for name in ['5', 'twoTH', 'ThreeEN']:
            sample = cv2.imread(name + '.jpg')
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            ret, sample = cv2.threshold(sample, 127, 255, 0)
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

        p_ret = cv2.matchShapes(img, LMG[0], 1, 0.0)
        for i in range(1, len(LMG)):
            ret = cv2.matchShapes(img, LMG[i], 1, 0.0)
            if ret < p_ret:
                p_ret = ret
        return ret

    def aspectRatio(self, img_f):
        img_fc = copy.deepcopy(img_f)
        img_fc, cfc, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            xfc, yfc, wfc, hfc = cv2.boundingRect(cfc[-1])
        except:
            return 1.0
        aspect_ratio = float(wfc) / hfc
        return aspect_ratio

    def getWordSize(self, img_f):
        img_fc = copy.deepcopy(img_f)
        img_fc, contours, hfc = cv2.findContours(img_fc, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        try:
            cnt = contours[-1]
        except:
            return 0, 0
        leftmost = np.array(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = np.array(cnt[cnt[:, :, 0].argmax()][0])
        topmost = np.array(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = np.array(cnt[cnt[:, :, 1].argmax()][0])
        return np.linalg.norm(leftmost - rightmost), np.linalg.norm(topmost - bottommost)

    def regress_to_real_world(self, points):
        if points != ():
            feature_x, feature_y = IP.get_XY_feature(points)
            new_x = self.model_x.predict([feature_x])
            new_y = self.model_y.predict([feature_y])
            if 'L' in self.name:
                return [-500 + self.z, new_x[0], new_y[0][0]]
            elif 'R' in self.name:
                return [500 - self.z, new_x[0], new_y[0][0]]
            elif 'B' in self.name:
                return [new_x[0], new_y[0][0], self.z]
        else:
            return ()

    def orientation_to_mat(self, orientation):
        if orientation is None:
            return None
        else:
            orientation = orientation+OFFSET_ORIENTATION
            if 'L' in self.name:
                return np.array([[0, 0, -1],
                                 [np.sin(orientation[0]), np.cos(orientation[0]), 0],
                                 [np.cos(orientation[0]), -np.sin(orientation[0]), 0]])
            elif 'R' in self.name:
                return np.array([[0, 0, 1],
                                 [np.sin(orientation[0]), np.cos(orientation[0]), 0],
                                 [-np.cos(orientation[0]), np.sin(orientation[0]), 0]])
            elif 'B' in self.name:
                return np.array([[np.cos(orientation[0]), -np.sin(orientation[0]), 0],
                                 [-np.sin(orientation[0]), -np.cos(orientation[0]), 0],
                                 [0, 0, -1]])

    def draw_x_vector(self,orientation_matrix):
        if 'L' in self.name:
            print(orientation_matrix[1:,0:2])
        elif 'R' in self.name:
            print(orientation_matrix[1:, 0:2])
        elif 'B' in self.name:
            print(orientation_matrix[1:, 0:2])

