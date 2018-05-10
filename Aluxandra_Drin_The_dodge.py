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
from math import degrees, radians, pi
from time import sleep

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
MANUAL_STEP = False

# main serial setting
RUN_MATLAB = False
CHECK_LASER = False
PATH_PLANING_MODE = 3 # 0 -> None , 1-> wan , 2 -> zumo , 3 -> combine
SERVO_PLANING = False
HALF_IK = False

# initial constant 
INITIAL_POSITION = [[0,300,700],'F',MAN.RE_F]
PLATE_POSITION_X = [-300,-100,100,300]
PLATE_POSITION_Y = 670
PLATE_POSITION_Z = [750,500,300]  
OFFSET_LENGHT_IN = 180  # before to position   
OFFSET_LENGHT_OUT_BOTTOM = 500 # after to position of bottom wall
OFFSET_LENGHT_OUT_OTHER = 240 # after to position of left and right wall
EXTRA_OFFSET_IN = 150 # before put
EXTRA_OFFSET_OUT = 150 # after put
PLATE_HEIGHT = 14
STEP_ROTATION = 4
STEP_DISTANCE = 100
STEP_OFFSET_DISTANCE = 10

#offset set Q
OFFSET_Q = [230,35,160 ,135,130,135] # [230,32,147,135,135,135]
GAIN_Q = [-1,1,1,1,1,1]
GAIN_MAGNETIC = 7/9

NEW_Z_EQUATION = lambda r: -0.08951*r + 47.72    # 0 if r > 375 else 0
Q_FOR_BACKLASH = [lambda x: 0, lambda x: x[1],lambda x: x[1]+x[2],lambda x: 0,
                        lambda x: 0,lambda x: 0 ]
OFFSET_BACKLASH = [lambda x: 0, lambda x: 3,lambda x: 5 ,lambda x: 0,
                        lambda x: 0,lambda x: 0 ]
CASE_BACKLASH = [lambda x:  radians(90), lambda x: radians(90), lambda x: radians(90)-x, lambda x: radians(135), 
                    lambda x: radians(135), lambda x: radians(135)]

ENLIGHT_POS = [[-200,400,700],[0,300,800],[200,400,700]]

# test condition
TEST_MODE = False
MODE_POSITION = True   
MODE_FIX_DATA = False 

# START STEP
POSITION_STEP_1 = [radians(90),radians(135),radians(-130),radians(0),radians(0),radians(0)]
POSITION_STEP_2 = [radians(90),radians(30),radians(-140),radians(0),radians(0),radians(0)] 
POSITION_STEP_3 = [radians(90),radians(135),radians(-130),radians(0),radians(0),radians(0)]

# test data
if TEST_MODE:
    if MODE_POSITION:
        # data = [ [[200,400,100],'B',10,MAN.RE_B], [[-400,200,800],'L',10,MAN.RE_L],[[400,200,700],'R',10,MAN.RE_R],  [[0,450,300],'B',10,MAN.RE_B]  ]
        # data = [ [[-200,400,700],'L',10,MAN.RE_L]]
        # data = [ [[200,650,600],'F',0,MAN.RE_F] ] 
        # data = [ [[300,650,500],'F',0,MAN.RE_F ] ] 
        # data = [ [[-300,400,100], 'B',0, MAN.RE_B ], [[-0,400,500], 'B',0, MAN.RE_B ] , [[300,400,500], 'B',0, MAN.RE_B ] ]
        # data  = [  [[300,600,700],'F',0,MAN.RE_F ], [[300,500,600],'F',0,MAN.RE_F ], [[300,500,400],'F',0,MAN.RE_F ], [[300,500,300],'F',0,MAN.RE_F ], [[300,500,200],'F',0,MAN.RE_F ]]
        # data = [  [[0,750,500], 'F',0, MAN.RE_F ] , [[-300,650,500],'F',0,MAN.RE_F ] , [[-300,650,700],'F',0,MAN.RE_F ] , [[-300,650,500],'F',0,MAN.RE_F ] , [[-300,650,700],'F',0,MAN.RE_F ]  ]
        # data = [  [[0,750,500], 'B',0, MAN.RE_B ] , [[-300,650,500],'B',0, MAN.RE_B  ] , [[-300,650,700],'B',0, MAN.RE_B ] , [[-300,650,500],'B',0, MAN.RE_B  ] , [[-300,650,700],'B',0, MAN.RE_B  ]  ]
        # data = [ [[-300,650,700],'F',0,MAN.RE_F ] ] 
        data = [[[0,350,25], 'B',0, MAN.RE_B ]]
    else:
       data= [radians(90),radians(30),radians(-140),radians(0),radians(0),radians(0)] 


# data = [95/180*pi,115/180*pi,-120/180*pi,135/180*pi,135/180*pi,135/180*pi]  # up rising
# data = [1/2*pi,0*pi,-110/180*pi,0*pi,0*pi,0*pi]                             # hua pak           
# data = [1/2*pi,-35/180*pi,-110/180*pi,0*pi,0*pi,0*pi]                       # drin the dodge
FIND_SAME_POINT_AND_AVERAGE_IT = False
LOAD_IMAGE = False
RADIANT = 75
ALL_PIC_PATH = {'L':'picture\\testpic\\TestLeftSide.jpg', 'R':'picture\\testpic\\TestRightSide.jpg', 'Bl':'picture\\testpic\\Bl_test1.png' , 
                'Bm':'picture\\testpic\\Bm_test1.png', 'Br':'picture\\testpic\\Br_test1.png' }
IMG_PATH = ALL_PIC_PATH['L']

ROTATION = 90
SAVE_IMG_NAME = 'BL_Bside.jpg'
FOUR_POINTS = np.array([[557, 202], [377, 197], [289, 595], [559, 638]])
IMAGE_SIZE = (32,64)
CAMERA_ALL_OFFSET_Z = 25

CAM_LEFT_PORT = 1
CAM_LEFT_MODE = 1
CAM_LEFT_ORIENTATION = 0
CAM_LEFT_FOUR_POINTS = np.array([[81, 0], [81, 639], [493, 530], [603, 85]])
CAM_LEFT_MINIMUM_AREA = 0.01
CAM_LEFT_MAXIMUM_AREA = 0.9
CAM_LEFT_LENGTH_PERCENT = 0.01
CAM_LEFT_THRESH_KERNEL = 21
CAM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_LEFT_THRESH_KERNEL_ORIGINAL = 21
CAM_LEFT_BOUNDARY = 20
CAM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_LEFT_CLOSING_KERNEL_SIZE = 5
CAM_LEFT_OFFSET_HOMO_X = -81#-300
CAM_LEFT_OFFSET_HOMO_Y = 0#-100

# Camera Right
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 0
CAM_RIGHT_FOUR_POINTS =np.array([[559, 4], [560, 639], [145, 504], [42, 77]])
CAM_RIGHT_MINIMUM_AREA = 0.01
CAM_RIGHT_MAXIMUM_AREA = 0.9
CAM_RIGHT_LENGTH_PERCENT = 0.01
CAM_RIGHT_THRESH_KERNEL = 21
CAM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_RIGHT_THRESH_KERNEL_ORIGINAL = 37
CAM_RIGHT_BOUNDARY = 20
CAM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_RIGHT_CLOSING_KERNEL_SIZE = 4
CAM_RIGHT_OFFSET_HOMO_X = -42#-300
CAM_RIGHT_OFFSET_HOMO_Y = 0#-100

# Camera Bottom Middle
CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = 0
CAM_BOTTOM_MIDDLE_FOUR_POINTS =  np.array([[17, 483], [178, 293], [485, 285], [637, 479]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.03
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 140
CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL = 140
CAM_BOTTOM_MIDDLE_BOUNDARY = 10
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = -2
CAM_BOTTOM_MIDDLE_CLOSING_KERNEL_SIZE =1
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -17
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = -285

# Camera Bottom Right
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = 0
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[113, 167], [308, 181], [356, 631], [68, 638]])
# np.array([[119, 173], [51, 639], [358, 638], [314, 189]])
# np.array([[302,237],[106,230],[65,638],[358,638]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])
CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.10
CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.001
CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
CAM_BOTTOM_RIGHT_BOUNDARY = 20
CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_RIGHT_CLOSING_KERNEL_SIZE = 5
CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68#-50
CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167#-100

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 0
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[348, 147], [547, 138], [549, 629], [269, 599]])
# np.array([[544, 173], [553, 638], [269, 621], [348, 182]])
# np.array([[342,145],[267,628],[554,639],[550,143]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])
CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_LEFT_THRESH_KERNEL = 21
CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL = 21
CAM_BOTTOM_LEFT_BOUNDARY = 20
CAM_BOTTOM_LEFT_CLOSING_KERNEL_SIZE = 3
CAM_BOTTOM_LEFT_BINARIZE_METHOD = -2
CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -269#-300
CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -138#-100


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
offsetQ = [205,35,150,0,0,0]
gainQ = [-1,1,1,1,1,1]



'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

# cam = Retinutella('cam1',1,0,cameraMode=1)
# cam1 = lambda x : Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS)
# cam2 = lambda x : Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)
# cam3 = lambda x : Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
#                             CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS)
# cam4 = lambda x : Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
#                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)
# cam5 = lambda x : Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
#                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT)


cam1 = lambda : Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS,
                       thresh_kernel=CAM_LEFT_THRESH_KERNEL,
                       thresh_kernel_original=CAM_LEFT_THRESH_KERNEL_ORIGINAL,
                       minimum_area=CAM_LEFT_MINIMUM_AREA
                       , minimum_area_original=CAM_LEFT_MINIMUM_AREA_ORIGINAL,
                       maximum_area=CAM_LEFT_MAXIMUM_AREA,
                       maximum_area_original=CAM_LEFT_MAXIMUM_AREA_ORIGINAL,
                       lengthpercent=CAM_LEFT_LENGTH_PERCENT,
                       lengthpercent_original=CAM_LEFT_LENGTH_PERCENT_ORIGINAL,
                       word_boundary=CAM_LEFT_BOUNDARY, binarize_method=CAM_LEFT_BINARIZE_METHOD,closing_kernel_size=CAM_LEFT_CLOSING_KERNEL_SIZE)

cam2 = lambda : Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z,
                        CAM_RIGHT_FOUR_POINTS,
                        thresh_kernel=CAM_RIGHT_THRESH_KERNEL,
                        thresh_kernel_original=CAM_RIGHT_THRESH_KERNEL_ORIGINAL,
                        minimum_area=CAM_RIGHT_MINIMUM_AREA
                        , minimum_area_original=CAM_RIGHT_MINIMUM_AREA_ORIGINAL,
                        maximum_area=CAM_RIGHT_MAXIMUM_AREA,
                        maximum_area_original=CAM_RIGHT_MAXIMUM_AREA_ORIGINAL,
                        lengthpercent=CAM_RIGHT_LENGTH_PERCENT,
                        lengthpercent_original=CAM_RIGHT_LENGTH_PERCENT_ORIGINAL,
                        word_boundary=CAM_RIGHT_BOUNDARY, binarize_method=CAM_RIGHT_BINARIZE_METHOD,closing_kernel_size=CAM_RIGHT_CLOSING_KERNEL_SIZE
                        )

cam3 = lambda : Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS,thresh_kernel=CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                            thresh_kernel_original=CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_MIDDLE_MINIMUM_AREA
                            ,minimum_area_original=CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                            maximum_area_original=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT,
                            lengthpercent_original=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL,word_boundary=CAM_BOTTOM_MIDDLE_BOUNDARY,binarize_method=CAM_BOTTOM_MIDDLE_BINARIZE_METHOD,
                                closing_kernel_size=CAM_BOTTOM_MIDDLE_CLOSING_KERNEL_SIZE)


cam4 = lambda : Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                               CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
                               CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT,thresh_kernel=CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                             thresh_kernel_original=CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_RIGHT_MINIMUM_AREA
                             ,minimum_area_original=CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                             maximum_area_original=CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_RIGHT_LENGTH_PERCENT,
                               Offset_homo_x=CAM_BOTTOM_RIGHT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y,closing_kernel_size=CAM_RIGHT_CLOSING_KERNEL_SIZE)


cam5 = lambda : Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT,thresh_kernel=CAM_BOTTOM_LEFT_THRESH_KERNEL,
                             thresh_kernel_original=CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_LEFT_MINIMUM_AREA
                             ,minimum_area_original=CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_LEFT_MAXIMUM_AREA,
                             maximum_area_original=CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_LEFT_LENGTH_PERCENT,
                              Offset_homo_x=CAM_BOTTOM_LEFT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_LEFT_OFFSET_HOMO_Y,closing_kernel_size=CAM_BOTTOM_LEFT_CLOSING_KERNEL_SIZE)

listCam = [ cam1,cam2,cam3,cam4, cam5]


send_serial = sendSerial(port=PORT, checkLaser = CHECK_LASER, runMatlab= RUN_MATLAB, sendSerial= SEND_SERIAL, manualStep= MANUAL_STEP, 
                pathPlaning = PATH_PLANING_MODE, initial_position = INITIAL_POSITION, recieveSerial= RECIEVE_SERIAL , extraoffsetIn= EXTRA_OFFSET_IN, 
                half_IK= HALF_IK, platePositionX= PLATE_POSITION_X, platePositionY = PLATE_POSITION_Y , servoPlaning= SERVO_PLANING, offsetLenghtOutOther= OFFSET_LENGHT_OUT_OTHER, 
                platePositionZ = PLATE_POSITION_Z, offsetLenghtIn=  OFFSET_LENGHT_IN, plateHeight = PLATE_HEIGHT, offsetLenghtOutBottom= OFFSET_LENGHT_OUT_BOTTOM,
                offsetQ= OFFSET_Q, gainQ = GAIN_Q ,modeFixData=MODE_FIX_DATA, stepRotation= STEP_ROTATION, stepOffsetDistance= STEP_OFFSET_DISTANCE, 
                enLightPos=ENLIGHT_POS, offsetBacklash = OFFSET_BACKLASH ,caseBacklash = CASE_BACKLASH, gainMagnetic= GAIN_MAGNETIC,
                qForBackLash= Q_FOR_BACKLASH, planingStepDistance= STEP_DISTANCE, extraoffsetOut= EXTRA_OFFSET_OUT, new_z_equation= NEW_Z_EQUATION)


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
*                    KNN model  0                   *
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

    # send_serial.getSetQAndWrite(POSITION_STEP_1,0)
    # input('press key to move to HIGH position')
    # send_serial.getSetQAndWrite(POSITION_STEP_2,0)
    # input('press key to move to LOW position')
    # send_serial.getSetQAndWrite(POSITION_STEP_3,0)
    # input('press key to move to HIGH position')
    oldCam = ''
    data = []
    all_image = []
    all_position = []
    all_orientation = []

        # get word from original picture
        # after pass through this section, we get list of word (image) in the variable
        # 'plate'.
    if LOAD_IMAGE == False:
        send_serial.getSetQAndWrite(POSITION_STEP_1,0)
    while len(all_image) == 0:
        for cam in listCam:
            cam = cam()
            IMG_PATH = ALL_PIC_PATH[cam.name]
            
            if LOAD_IMAGE == False:
                if cam.name in ['L','R'] and oldCam not in ['L','R']:
                    # input('press key to move to LOW position')
                    send_serial.getSetQAndWrite(POSITION_STEP_2,0)
                    

                elif cam.name in ['Bl','Bm','Br'] and oldCam not in ['Bl','Bm','Br'] :
                    # input('press key to move to HIGH position')
                    send_serial.getSetQAndWrite(POSITION_STEP_1,0)
                    
            oldCam = cam.name

            org,plate,platePos,plateOrientation = cam.getListOfPlate(image_size=IMAGE_SIZE, platePos=True,
                                                                            plateOrientation=True, show=True,
                                                                            LOAD_IMAGE=LOAD_IMAGE ,
                                                                            FILENAME=IMG_PATH)

            plate, platePos, plateOrient8ation = IP.filter_plate(plate_img = plate, plate_position = platePos, plate_Orientation= plateOrientation)
            # print('--------------')
            # # print(platePos)
            # # print(plateOrientation)
            # print('--------------')

            # check if plate is found
            if platePos != []:

                if () in  platePos :
                    CONTINUE

                for selectPlate in plate:
                    all_image.append(selectPlate)
                
                for selectPosition in platePos:
                    all_position.append(selectPosition)
                
                for selectOreintation in plateOrientation:
                    all_orientation.append(selectOreintation)

            # input(cam.name)
            if LOAD_IMAGE == False:
                cam.cam.release()   


    if FIND_SAME_POINT_AND_AVERAGE_IT:
        plate, platePos, plateOrientation = IP.find_same_point_and_average_it(all_image,all_position,all_orientation,r=RADIANT )# new_image,new_position,new_orientation
    else :
        plate = all_image
        platePos = all_position
        plateOrientation = all_orientation

    for selectPlate, selectPosition, selectOreintation in zip(plate, platePos, plateOrientation):
        
        if str(selectOreintation[0:3,2]) == '[ 0.  0. -1.]':
            wall = 'B'
        elif str(selectOreintation[0:3,2]) == '[-1.  0.  0.]':
            wall = 'L'
        elif str(selectOreintation[0:3,2]) == '[1. 0. 0.]':
            wall = 'R'

        data.append( [selectPosition, wall, mainPredict([selectPlate],model=MODEL)[0] ,selectOreintation] )

send_serial.getSetQAndWrite(POSITION_STEP_1,0)
print(data)
input('data from detection :')

if len(data) > 10:
    data = data[:10]
if MODE_POSITION or TEST_MODE == False:
    # data = position wall predict orentation
        
    send_serial.getXYZAndWrite(data)
else :
    send_serial.getSetQAndWrite(data,0)
        

# input("press some key and enter to exit : ")


    # #show and finally destroy those windows.
    # for p in range(0,len(plate)):
    #     plate[p] = cv2.resize(plate[p],(IMAGE_SIZE[1]*5,IMAGE_SIZE[0]*5))
    #     cam.show(plate[p],frame='plate_'+str(NUM2WORD[pred_result[p]]))
    #     cv2.moveWindow('plate_'+str(NUM2WORD[pred_result[p]]), 700,300);
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(corg, str(NUM2WORD[pred_result[p]]), (50, 400), font, 5, (0, 0, 255), 5, cv2.LINE_AA)
    # cam.show(corg,wait=30)
    # cam.destroyWindows()
