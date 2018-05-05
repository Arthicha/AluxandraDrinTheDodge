__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'main program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import copy
import math
# 1. system module
import os
import random
import sys
from math import degrees, pi

# 6. image processing module
import cv2
# 5. visualization module
import matplotlib.pyplot as plt
# 3. mathematical module
import numpy as np
import serial
# 2. machine learning module
import tensorflow as tf
from sklearn.externals import joblib

from module import PaZum
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
# serial and manipulator kinematics
from module.MANipulatorKinematics import MANipulator
from module.prePackage import prePackage
from module.PuenBan_K_Tua import PuenBan_K_Tua
from module.RandomFunction import *
from module.Retinutella_theRobotEye import Retinutella
from module.sendSerial import sendSerial
from module.serialCommu import serial_commu
# 4. our own module
from module.Tenzor import TenzorAE, TenzorCNN, TenzorNN
from module.UniqueCameraClass import *
from module.Zkeleton import Zkele

'''*************************************************
*                                                  *
*                  control variable                *
*                                                  *
*************************************************'''

# define augmentation mode
AUG_NONE = 0
AUG_DTSX = 1
AUG_DTSY = 2
AUG_DTSB = 3
AUG_LINX = 4
AUG_LINY = 5
AUG_LINB = 6

# machine learning model CNN, KNN, RF and HAR, ALL
ML_CNN = 0
ML_KNN = 1
ML_RF = 2
ML_HAR = 3
ML_ALL = 4

PATH = os.getcwd()
sep = os.path.sep

'''*************************************************
*                                                  *
*               configuration variable             *
*                                                  *
*************************************************'''
MAN = MANipulator()

# protocal serial setting
PORT = 4
SEND_SERIAL = True
RECIEVE_SERIAL = SEND_SERIAL

# main serial setting
RUN_MATLAB = False
CHECK_LASER = False
PATH_PLANING = False
SERVO_PLANING = False
HALF_IK = False

# initial constant 
INITIAL_POSITION = [[0,400,600],'F',MAN.RE_F]
PLATE_POSITION_X = [-300,-100,100,300]
PLATE_POSITION_Y = 600
PLATE_POSITION_Z = [700,500,300]  
OFSET_LENGHT = 60  # before to position
OFSET_LENGHT_2 = 120 # after to position
PLATE_HEIGHT = 50
WORKSPACE = [-500,500,-400,600,0,1000]
STEP_ROTATION = 8

#ofset set Q
OFSET_Q = [230,30,150,135,135,135] # [230,32,147,135,135,135]
GAIN_Q = [-1,1,1,1,1,1]

# test condition
TEST_MODE = True
MODE_POSITION = True
MODE_FIX_DATA = True


# test data
if TEST_MODE:
    if MODE_POSITION:
        # data = [ [[200,400,100],'B',0,MAN.RE_B], [[-400,200,800],'L',1,MAN.RE_L] ]
        data = [ [[0,450,300],'B',0,MAN.RE_B] ] 
        # daat = [[0,],'F',0,MAN.RE_F]
    else:
       data= [93/180*pi,90/180*pi,-100/180*pi,-90/180*pi,-90/180*pi,-90/180*pi]


# data = [95/180*pi,115/180*pi,-120/180*pi,135/180*pi,135/180*pi,135/180*pi]  # up rising
# data = [1/2*pi,0*pi,-110/180*pi,0*pi,0*pi,0*pi]                             # hua pak           
# data = [1/2*pi,-35/180*pi,-110/180*pi,0*pi,0*pi,0*pi]                       # drin the dodge

CAMERA_ALL_OFFSET_Z = 25

# Camera Left
CAM_LEFT_PORT = 1
CAM_LEFT_MODE = 1
CAM_LEFT_ORIENTATION = -90
CAM_LEFT_FOUR_POINTS = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Right
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = -90
CAM_RIGHT_FOUR_POINTS = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Bottom Middle
CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -90
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Bottom Right
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[72,638],[273,232],[93,227],[333,639]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 1
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[335,193],[538,182],[587,638],[294,639]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[0, 0], [80, 640], [0, 640], [80, 0]])



IMAGE_SIZE = (32,64)
N_CLASS = 30

# select machine learning model
MODEL = ML_CNN

# restore save model
# for example, PATH+"\\savedModel\\modelCNN"
GETT_CNN_PATH = PATH+"\\savedModel\\modelCNN_skele"
GETT_KNN_PATH = PATH+sep+'savedModel'+sep+'modelKNN'
GETT_RF_PATH = PATH+"\\savedModel\\modelRandomForest\\Random_Forest_best_run.gz"
GETT_HAR_PATH = PATH

CONTINUE = False
AUG_VALUE = [20,3]
MAGNIFY = [90,110]
MORPH = [1,5]
MOVE = [-3,3]

# convolutional neural network config
CNN_HIDDEN_LAYER = [48,64,128]
KERNEL_SIZE = [[5,5],[3,3]]
POOL_SIZE = [[4,4],[2,2]]
STRIDE_SIZE = [4,2]


# set numpy to print/show all every element in matrix
np.set_printoptions(threshold=np.inf)

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# control config
ofsetQ = [205,35,150,0,0,0]
gainQ = [-1,1,1,1,1,1]



'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

# cam = Retinutella('cam1',1,0,cameraMode=1)
# cam1 = Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS)
# cam2 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)
# cam3 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
#                             CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS)
# cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
#                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)
cam5 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT)

listCam = [cam5,cam5,cam5,cam5,cam5]


send_serial = sendSerial(port=PORT, checkLaser = CHECK_LASER, runMatlab= RUN_MATLAB, sendSerial= SEND_SERIAL,
                pathPlaning = PATH_PLANING, initial_position = INITIAL_POSITION, recieveSerial= RECIEVE_SERIAL ,
                half_IK= HALF_IK, platePositionX= PLATE_POSITION_X, platePositionY = PLATE_POSITION_Y , servoPlaning= SERVO_PLANING, 
                platePositionZ = PLATE_POSITION_Z, ofsetLenght = OFSET_LENGHT, plateHeight = PLATE_HEIGHT, ofsetLenght2= OFSET_LENGHT_2,
                workspace = WORKSPACE, ofsetQ = OFSET_Q, gainQ = GAIN_Q ,modeFixData=MODE_FIX_DATA, stepRotation= STEP_ROTATION)


NUM2WORD = ["0","1","2","3","4","5","6","7","8","9",
            "zero","one","two","three","four","five","six","seven","eight","nine",
            "soon","nung","song","sam","see","ha","hok","jed","pad","kaow"]

'''*************************************************
*                                                  *
*                 function                         *
*                                                  *
*************************************************'''
def log_prob(prob_1,prob_2,prob_3):
    sum_of_all = list(map(lambda x,y,z: np.add(np.add(x,y),z),prob_1,prob_2,prob_3))
    class_of_all = list(map(lambda x: np.argmax(x),sum_of_all))
    return class_of_all

'''*************************************************
*                                                  *
*                 setup program                    *
*                                                  *
*************************************************'''

if MODEL not in range(0,5):
    sys.stderr.write('MODEL ERROR: ',MODEL)
    sys.exit(-1)

'''*************************************************
*                                                  *
*                    CNN model                     *
*                                                  *
*************************************************'''

# create input section
with tf.name_scope('input_placeholder'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0]*IMAGE_SIZE[1]],name='x_data')
    y_ = tf.placeholder(tf.float32, shape=[None, N_CLASS],name='y_data')
    x_image = tf.reshape(x, [-1, IMAGE_SIZE[0],IMAGE_SIZE[1], 1])

# create model of convolutional neural network
with tf.name_scope('CNN_model'):
    CNN = TenzorCNN()
    y_pred,activity = CNN.CNN2(x_image,CNN_HIDDEN_LAYER,KERNEL_SIZE,POOL_SIZE,STRIDE_SIZE,IMAGE_SIZE)

# output and evaluation of the model
with tf.name_scope('evaluation'):
    pred_prob = tf.argmax(y_pred, 1)
    correct_prediction = tf.equal(pred_prob, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if GETT_CNN_PATH != None:
        saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if GETT_CNN_PATH != None:
    saver.restore(sess, GETT_CNN_PATH+'\\modelCNN.ckpt')
    print("Get model from path: %s" % GETT_CNN_PATH+'\\modelCNN.ckpt')

'''*************************************************
*                                                  *
*                    KNN model                     *
*                                                  *
*************************************************'''

testKNN = PuenBan_K_Tua()

def predictKNN(img=[]):
    hog = testKNN.HOG_int()

    test_hog_descriptors = []
    rePred = []

    for imgCount in range(len(img)):
    # data = np.random.sample((32*64)) *255
        data = img[imgCount].astype(np.uint8)

        data = data.reshape(-1,(64))
        imgs = data.astype(np.uint8)

        imgs = testKNN.deskew(imgs)
        try :
            test_hog_descriptors.append(hog.compute(imgs,winStride=(16,16)))
            test_hog_descriptors.append(hog.compute(imgs,winStride=(16,16)))

            test_hog_descriptors = np.squeeze(test_hog_descriptors)
            model = joblib.load(GETT_KNN_PATH+ sep+ 'knn_model_real.pkl','r')

            pred = model.predict(test_hog_descriptors.tolist())
            rePred.append(int(pred[0]))
        except:
            rePred.append(int(pred[0]))
    return rePred

def mainPredict(plate,model='ALL'):
    # prediction section
    # pred_result store predicted result from spicified machine learning model.
    pred_result = np.zeros(len(plate))
    list_vector = np.zeros(len(plate))
    plate2show = copy.deepcopy(plate)

    # preprocessing image
    for p in range(0,len(plate)):
        #plate[p] = IP.auto_canny(plate[p])
        #plate[p] = 255 - plate[p]
        plate[p] = Zkele(plate[p],method='3d')
    if plate != []:
        # preparing input, convert image to vector
        list_vector = np.resize(np.array(plate),(len(plate),IMAGE_SIZE[0]*IMAGE_SIZE[1]))
        # convert 8 bit image to be in range of 0.00-1.00, dtype = float
        list_vector = list_vector/255
        pred_result1 = sess.run(y_pred,feed_dict={x: list_vector})
        pred_result2 = predictKNN(list_vector*255)
        pred_result3 = forest.predict(plate,'prob')
        pred_result=log_prob(pred_result1,pred_result2,pred_result3)

        if model == ML_CNN:
            output_result = pred_result
        elif model == ML_KNN:
            output_result = pred_result1
        elif model == ML_RF:
            output_result = pred_result2
        elif model == ML_ALL:
            output_result = pred_result3
        return output_result

'''*************************************************
*                                                  *
*             Random Forest model                  *
*                                                  *
*************************************************'''
if 1:
    if GETT_RF_PATH != None:
        forest = PaZum.PaZum(GETT_RF_PATH)
    else:
        raise SyntaxError(" If u give us no path to the model how could we restore it!!!")

'''*************************************************
*                                                  *
*                   main program                   *
*                                                  *
*************************************************'''

# main loop


if TEST_MODE == False:
    data = []
    all_image = []
    all_position = []
    all_orientation = []
    
        # get word from original picture
        # after pass through this section, we get list of word (image) in the variable
        # 'plate'.
    while len(all_image) == 0:
        for cam in listCam:
                org,plate,platePos,plateOrientation = cam.getListOfPlate(image_size=IMAGE_SIZE,platePos= True,show= True,plateOrientation= True)

                plate, platePos, plateOrientation = IP.filter_plate(plate_img = plate, plate_position = platePos, plate_Orientation= plateOrientation)
                # print('--------------')
                # # print(platePos)
                # # print(plateOrientation)
                # print('--------------')

                # check if plate is found
                if platePos != []:

                    if () in  platePos :
                        CONTINUE

                    # convert pos wrt robot base
                    # tew -> set the wrong frame x_tew = y, y_tew = x, z -> same
                    # x = -(platePos[0][1]) -500    
                    # y = platePos[0][0] - 300
                    # z = platePos[0][2]
                    # print((x,y,z))
                    # decoupling

                    wall = (str(cam.name)[0]).upper()

                    for selectPlate in plate:
                        all_image.append(selectPlate)
                    
                    for selectPosition in platePos:
                        all_position.append(selectPosition)
                    
                    for selectOreintation in plateOrientation:
                        all_orientation.append(selectOreintation)


        plate, platePos, plateOrientation = IP.find_same_point_and_average_it(all_image,all_position,all_orientation,r=75)# new_image,new_position,new_orientation

    for selectPlate, selectPosition, selectOreintation in zip(plate, platePos, plateOrientation):
        print(str(selectOreintation[0:3,2]))
        if str(selectOreintation[0:3,2]) == '[0. 0. -1.]':
            wall = 'B'
        elif str(selectOreintation[0:3,2]) == '[-1. 0. 0.]':
            wall = 'L'
        elif str(selectOreintation[0:3,2]) == '[1. 0. 0.]':
            wall = 'R'
        data.append( [selectPosition, wall, mainPredict([selectPlate],model=MODEL)[0] ,selectOreintation] )

# print(data)

if len(data) > 10:
    data = data[:10]
if MODE_POSITION or (TEST_MODE == False) :
    # data = position wall predict orentation
    send_serial.getXYZAndWrite(data)
else :
    send_serial.getSetQAndWrite(data,00)
        

input("press some key and enter to exit")




    # #show and finally destroy those windows.
    # for p in range(0,len(plate)):
    #     plate[p] = cv2.resize(plate[p],(IMAGE_SIZE[1]*5,IMAGE_SIZE[0]*5))
    #     cam.show(plate[p],frame='plate_'+str(NUM2WORD[pred_result[p]]))
    #     cv2.moveWindow('plate_'+str(NUM2WORD[pred_result[p]]), 700,300);
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(corg, str(NUM2WORD[pred_result[p]]), (50, 400), font, 5, (0, 0, 255), 5, cv2.LINE_AA)
    # cam.show(corg,wait=30)
    # cam.destroyWindows()
