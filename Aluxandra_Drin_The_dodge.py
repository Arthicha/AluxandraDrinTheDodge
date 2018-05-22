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
SEND_SERIAL = True # send serial package to board
RECIEVE_SERIAL = SEND_SERIAL
MANUAL_STEP = False # press key all round
KEEP_DATA = False # open save position mode

# main serial setting
RUN_MATLAB = False # same as name / don't open it
CHECK_LASER = False # same as name / don't open it
PATH_PLANING_MODE = 3 # 0 -> None , 1-> wan , 2 -> zumo , 3 -> combine
SERVO_PLANING = False # servo sub move generate
HALF_IK = False # doesn't change it to True
SIMULATOR = False # doesn't open it when real run

# initial constant 
INITIAL_POSITION = [[0,350,700],'F',MAN.RE_F]
PLATE_POSITION_X = [-260,-90,90,260]
PLATE_POSITION_Y = 645
PLATE_POSITION_Z = [725,550,300]

OFFSET_LENGHT_IN = 80  # before to position   
OFFSET_LENGHT_OUT_BOTTOM = 500 # after to position of bottom wall
OFFSET_LENGHT_OUT_OTHER = 80 # after to position of left and right wall
EXTRA_OFFSET_IN = 150 # before put
EXTRA_OFFSET_OUT = 150 # after put
PLATE_HEIGHT = 14
STEP_ROTATION = 4
STEP_DISTANCE = 25
STEP_OFFSET_DISTANCE = 10

#offset set Q
OFFSET_Q = [230,35,160 ,135,135,135] # [230,32,147,135,135,135]
GAIN_Q = [-1,1,1,1,1,1]
GAIN_MAGNETIC = 7/9

NEW_Z_EQUATION = lambda r: -0.08951*r + 47.72    # 0 if r > 375 else 0

OFFSET_BACKLASH = [lambda x: 0, lambda x: 5,lambda x: 10 ,lambda x: 0,
                        lambda x: 0,lambda x: 0 ]
CASE_BACKLASH = [lambda x:  radians(90), lambda x: radians(90), lambda x: radians(90)-x[1], lambda x: radians(135), 
                    lambda x: radians(135), lambda x: radians(135)]

ENLIGHT_POS = [[-200,400,700],[0,350,700],[200,400,700]]

# test condition
TEST_MODE = False
MODE_POSITION = True
MODE_FIX_DATA = False

# START STEP
POSITION_STEP_1 = [radians(90),radians(135),radians(-130),radians(0),radians(0),radians(0)]
POSITION_STEP_2 = [radians(90),radians(30),radians(-140),radians(0),radians(0),radians(0)] 

HOME_STEP1 = [radians(-10),radians(105),radians(-150),radians(0),radians(-90),radians(0)]
HOME_STEP15 = [radians(45),radians(110),radians(-150),radians(0),radians(-90),radians(0)]
HOME_STEP2 = [radians(90),radians(135),radians(-150),radians(0),radians(-90),radians(0)]


# test data
if TEST_MODE:
    if MODE_POSITION:
        # data = [ [[-300,400,100], 'B',0, MAN.RE_B ], [[-0,400,500], 'B',0, MAN.RE_B ] , [[300,400,500], 'B',0, MAN.RE_B ] ]
        # data  = [  [[300,600,700],'F',0,MAN.RE_F ], [[300,500,600],'F',0,MAN.RE_F ], [[300,500,400],'F',0,MAN.RE_F ], [[300,500,300],'F',0,MAN.RE_F ], [[300,500,200],'F',0,MAN.RE_F ]]
        # data = [  [[0,750,500], 'F',0, MAN.RE_F ] , [[-300,650,500],'F',0,MAN.RE_F ] , [[-300,650,700],'F',0,MAN.RE_F ] , [[-300,650,500],'F',0,MAN.RE_F ] , [[-300,650,700],'F',0,MAN.RE_F ]  ]
        # data = [  [[0,750,500], 'B',0, MAN.RE_B ] , [[-300,650,500],'B',0, MAN.RE_B  ] , [[-300,650,700],'B',0, MAN.RE_B ] , [[-300,650,500],'B',0, MAN.RE_B  ] , [[-300,650,700],'B',0, MAN.RE_B  ]  ]
        # data = [ [[-300,650,500],'F',0,MAN.RE_F ] ] 
        # data = [[[0,350,25], 'B',0, MAN.RE_B ]]
        # data = [ [[-425,600,700] ,'L',17,MAN.RE_L ], [[425,600,700] ,'R',10,MAN.RE_R ], [[371,392,25] ,'B',17,MAN.RE_B ]  ]
        # data = [[[-500,565,700] ,'L',17,MAN.RE_L ] ]
        # data = [ [ [-200,425,667],'B',0,MAN.RE_B ] ]
        data = [ [[-475.0, 551.2993603576629, 630.5625787777094], 'L', 0, np.array([[ 0.        ,  0.        , -1.        ],
       [ 0.20582585,  0.97858864,  0.        ],
       [ 0.97858864, -0.20582585,  0.        ]])], [[-475.0, 245.74050512504883, 710.701509479417], 'L', 19, np.array([[ 0.        ,  0.        , -1.        ],
       [ 0.54862332,  0.83606964,  0.        ],
       [ 0.83606964, -0.54862332,  0.        ]])], [[475.0, 328.4786052588786, 393.4999013780226], 'R', 18, np.array([[ 0.        ,  0.        ,  1.        ],
       [-0.11519377,  0.99334306,  0.        ],
       [-0.99334306, -0.11519377,  0.        ]])], [[475.0, 180.27426381337818, 711.5402149226298], 'R', 7, np.array([[ 0.        ,  0.        ,  1.        ],
       [ 0.47770828,  0.87851852,  0.        ],
       [-0.87851852,  0.47770828,  0.        ]])], [[475.0, 474.41021880213293, 778.7605630040975], 'R', 15, np.array([[ 0.        ,  0.        ,  1.        ],
       [ 0.05712606,  0.99836695,  0.        ],
       [-0.99836695,  0.05712606,  0.        ]])], [[377.86777357142137, 115.7263742812928, 25.0], 'B', 8, np.array([[ 0.96794957, -0.25114468,  0.        ],
       [-0.25114468, -0.96794957,  0.        ],
       [ 0.        ,  0.        , -1.        ]])], [[279.1847441565908, 498.7037793725063, 25], 'B', 28, np.array([[ 0.98133159, -0.19232339,  0.        ],
       [-0.19232339, -0.98133159,  0.        ],
       [ 0.        ,  0.        , -1.        ]])], [[-270.2495874121164, 521.6234534714949, 25], 'B', 25, np.array([[ 0.89717662, -0.441672  ,  0.        ],
       [-0.441672  , -0.89717662,  0.        ],
       [ 0.        ,  0.        , -1.        ]])], [[-366.76389939798526, 124.55950482568153, 25.0], 'B', 8, np.array([[ 0.97000611, -0.24308044,  0.        ],
       [-0.24308044, -0.97000611,  0.        ],
       [ 0.        ,  0.        , -1.        ]])]]

    #     data = [ [[-590.0, 651.2993603576629, 730.5625787777094], 'L', 0, np.array([[ 0.        ,  0.        , -1.        ],
    #    [ 0.20582585,  0.97858864,  0.        ],
    #    [ 0.97858864, -0.20582585,  0.        ]])] ]

    #     data = [  [[377.86777357142137, 115.7263742812928, 25.0], 'B', 8, np.array([[ 0.96794957, -0.25114468,  0.        ],
    #    [-0.25114468, -0.96794957,  0.        ],
    #    [ 0.        ,  0.        , -1.        ]])], [[279.1847441565908, 498.7037793725063, 25], 'B', 28, np.array([[ 0.98133159, -0.19232339,  0.        ],
    #    [-0.19232339, -0.98133159,  0.        ],
    #    [ 0.        ,  0.        , -1.        ]])], [[-270.2495874121164, 521.6234534714949, 25], 'B', 25, np.array([[ 0.89717662, -0.441672  ,  0.        ],
    #    [-0.441672  , -0.89717662,  0.        ],
    #    [ 0.        ,  0.        , -1.        ]])], [[-366.76389939798526, 124.55950482568153, 25.0], 'B', 8, np.array([[ 0.97000611, -0.24308044,  0.        ],
    #    [-0.24308044, -0.97000611,  0.        ],
    #    [ 0.        ,  0.        , -1.        ]])]]
       
        # data = [[[-600,565,700] ,'L',17,MAN.RE_L ] ]

    else:
    #    data= [radians(-10),radians(105),radians(-150),radians(0),radians(-90),radians(0)]
       data = [radians(90),radians(135),radians(-150),radians(0),radians(0),radians(0)]
    #    data = [radians(90),radians(45),radians(-150),radians(0),radians(0),radians(0)]


# data = [95/180*pi,115/180*pi,-120/180*pi,135/180*pi,135/180*pi,135/180*pi]  # up rising
# data = [1/2*pi,0*pi,-110/180*pi,0*pi,0*pi,0*pi]                             # hua pak           
# data = [1/2*pi,-35/180*pi,-110/180*pi,0*pi,0*pi,0*pi]                       # drin the dodge

FIND_SAME_POINT_AND_AVERAGE_IT = False
LOAD_IMAGE = False
RADIANT = 75
ALL_PIC_PATH = {'L':'picture\\testpic\\TestLeftSide.jpg', 'R':'picture\\testpic\\TestRightSide.jpg', 'Bl_bottom':'picture\\testpic\\Bl_test1.png' , 
                'Bm':'picture\\testpic\\Bm_test1.png', 'Br_bottom':'picture\\testpic\\Br_test1.png' }
IMG_PATH = ALL_PIC_PATH['L']

ROTATION = 90
SAVE_IMG_NAME = 'BL_Bside.jpg'
FOUR_POINTS = np.array([[557, 202], [377, 197], [289, 595], [559, 638]])
IMAGE_SIZE = (32,64)
CAMERA_ALL_OFFSET_Z = 25

CAM_LEFT_NAME = 'L'
CAM_LEFT_PORT = 0
CAM_LEFT_MODE = 1
CAM_LEFT_ORIENTATION = -90
CAM_LEFT_FOUR_POINTS = np.array([[82, 1], [75, 639], [492, 520], [600, 80]])
CAM_LEFT_MINIMUM_AREA = 0.01
CAM_LEFT_MAXIMUM_AREA = 0.9
CAM_LEFT_LENGTH_PERCENT = 0.01
CAM_LEFT_THRESH_KERNEL = 21
CAM_LEFT_BOUNDARY = 20
CAM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_LEFT_OFFSET_HOMO_X = -81  # -300
CAM_LEFT_OFFSET_HOMO_Y = 0  # -100

CAM_BOTTOM_MIDDLE_NAME = 'Bm'
CAM_BOTTOM_MIDDLE_PORT = 1
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -180
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[0, 433], [136, 157], [500, 161], [639, 422]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.01
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 37
CAM_BOTTOM_MIDDLE_BOUNDARY = 5
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -42  # -300
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = 0  # -100

# Camera Right
CAM_RIGHT_NAME = 'R'
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 90
CAM_RIGHT_FOUR_POINTS = np.array([[559, 4], [560, 639], [145, 504], [42, 77]])
CAM_RIGHT_MINIMUM_AREA = 0.01
CAM_RIGHT_MAXIMUM_AREA = 0.9
CAM_RIGHT_LENGTH_PERCENT = 0.01
CAM_RIGHT_THRESH_KERNEL = 21
CAM_RIGHT_BOUNDARY = 20
CAM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_RIGHT_OFFSET_HOMO_X = -42  # -300
CAM_RIGHT_OFFSET_HOMO_Y = 0  # -100

# Camera Bottom Right
CAM_BOTTOM_RIGHT_NAME = 'Br_bottom'
CAM_BOTTOM_RIGHT_PORT = 3
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS = np.array([[113, 167], [308, 181], [356, 631], [68, 638]])
CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
CAM_BOTTOM_RIGHT_BOUNDARY = 20
CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68  # -50
CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167  # -100

# Camera Bottom Left
CAM_BOTTOM_LEFT_NAME = 'Bl_bottom'
CAM_BOTTOM_LEFT_PORT = 4
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS = np.array([[348, 147], [547, 138], [549, 629], [269, 599]])
CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.87
CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.08
CAM_BOTTOM_LEFT_THRESH_KERNEL = 37
CAM_BOTTOM_LEFT_BOUNDARY = 20
CAM_BOTTOM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -269  # -300
CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -138  # -100


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


cam1 = lambda : Retinutella(CAM_LEFT_NAME, 0, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAM_LEFT_FOUR_POINTS,
                       CAM_LEFT_THRESH_KERNEL, CAM_LEFT_MINIMUM_AREA, CAM_LEFT_MAXIMUM_AREA, CAM_LEFT_LENGTH_PERCENT,
                       CAM_LEFT_BOUNDARY, CAM_LEFT_BINARIZE_METHOD)
cam2 = lambda : Retinutella(CAM_RIGHT_NAME, CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAM_RIGHT_FOUR_POINTS,
                        CAM_RIGHT_THRESH_KERNEL, CAM_RIGHT_MINIMUM_AREA, CAM_RIGHT_MAXIMUM_AREA,
                        CAM_RIGHT_LENGTH_PERCENT, CAM_RIGHT_BOUNDARY, CAM_RIGHT_BINARIZE_METHOD)


cam3 = lambda : Retinutella(CAM_BOTTOM_MIDDLE_NAME, CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION,
                                CAM_BOTTOM_MIDDLE_MODE, CAM_BOTTOM_MIDDLE_FOUR_POINTS, CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                                CAM_BOTTOM_MIDDLE_MINIMUM_AREA, CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                                CAM_BOTTOM_MIDDLE_LENGTH_PERCENT, CAM_BOTTOM_MIDDLE_BOUNDARY,
                                CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)


cam4 = lambda : Retinutella(CAM_BOTTOM_RIGHT_NAME, 0, CAM_BOTTOM_RIGHT_ORIENTATION,
                               CAM_BOTTOM_RIGHT_MODE, CAM_BOTTOM_RIGHT_FOUR_POINTS, CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                               CAM_BOTTOM_RIGHT_MINIMUM_AREA, CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                               CAM_BOTTOM_RIGHT_LENGTH_PERCENT, CAM_BOTTOM_RIGHT_BOUNDARY,
                               CAM_BOTTOM_RIGHT_BINARIZE_METHOD)

cam5 = lambda : Retinutella(CAM_BOTTOM_LEFT_NAME, CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION,
                              CAM_BOTTOM_LEFT_MODE, CAM_BOTTOM_LEFT_FOUR_POINTS, CAM_BOTTOM_LEFT_THRESH_KERNEL,
                              CAM_BOTTOM_LEFT_MINIMUM_AREA, CAM_BOTTOM_LEFT_MAXIMUM_AREA,
                              CAM_BOTTOM_LEFT_LENGTH_PERCENT, CAM_BOTTOM_LEFT_BOUNDARY, CAM_BOTTOM_LEFT_BINARIZE_METHOD)

# listCam = [ cam1,cam2,cam3,cam4, cam5]
listCam = [cam1]


send_serial = sendSerial(port=PORT, checkLaser = CHECK_LASER, runMatlab= RUN_MATLAB, sendSerial= SEND_SERIAL, manualStep= MANUAL_STEP, 
                pathPlaning = PATH_PLANING_MODE, initial_position = INITIAL_POSITION, recieveSerial= RECIEVE_SERIAL , extraoffsetIn= EXTRA_OFFSET_IN, 
                half_IK= HALF_IK, platePositionX= PLATE_POSITION_X, platePositionY = PLATE_POSITION_Y , servoPlaning= SERVO_PLANING, offsetLenghtOutOther= OFFSET_LENGHT_OUT_OTHER, 
                platePositionZ = PLATE_POSITION_Z, offsetLenghtIn=  OFFSET_LENGHT_IN, plateHeight = PLATE_HEIGHT, offsetLenghtOutBottom= OFFSET_LENGHT_OUT_BOTTOM,
                offsetQ= OFFSET_Q, gainQ = GAIN_Q ,modeFixData=MODE_FIX_DATA, stepRotation= STEP_ROTATION, stepOffsetDistance= STEP_OFFSET_DISTANCE, 
                enLightPos=ENLIGHT_POS, offsetBacklash = OFFSET_BACKLASH ,caseBacklash = CASE_BACKLASH, gainMagnetic= GAIN_MAGNETIC,
                planingStepDistance= STEP_DISTANCE, extraoffsetOut= EXTRA_OFFSET_OUT, new_z_equation= NEW_Z_EQUATION, simulator= SIMULATOR, keepData=KEEP_DATA)


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

        if model == 'CNN':
            output_result = pred_result
        elif model == 'KNN':
            output_result = pred_result1
        elif model == 'RF':
            output_result = pred_result2
        elif model == 'ALL':
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
    send_serial.getSetQAndWrite(HOME_STEP15,0)
    send_serial.getSetQAndWrite(HOME_STEP1,0)
    input('start program')
    send_serial.getSetQAndWrite(HOME_STEP15,0)
    send_serial.getSetQAndWrite(POSITION_STEP_1,0)
    oldCam = ''
    data = []
    all_image = []
    all_position = []
    all_orientation = []

        # get word from original picture
        # after pass through this section, we get list of word (image) in the variable
        # 'plate'.
    while len(all_image) == 0:
        for cam in listCam:
            cam = cam()
            IMG_PATH = ALL_PIC_PATH[cam.name]
            sleep(1)
            
            if LOAD_IMAGE == False:
                if cam.name in ['L','R'] :
                    # move to LOW position
                    send_serial.getSetQAndWrite(POSITION_STEP_2,0)

                elif cam.name in ['Bl_bottom','Bm','Br_bottom']  :
                    # move to HIGH position
                    send_serial.getSetQAndWrite(POSITION_STEP_1,0)

            # sleep(3)
            ret,_ = cam.cam.read()        
            oldCam = cam.name

            org,plate,platePos,plateOrientation = cam.getListOfPlate(image_size=IMAGE_SIZE,
                                                                            show=True,
                                                                            LOAD_IMAGE=LOAD_IMAGE ,
                                                                            FILENAME=IMG_PATH)

            if cam.name == 'Br' or cam.name=='Bl':
                b1,b2,b3,_,__ = cam.getImage(remove_pers=True)
            else:
                b1,b2,b3 = cam.getImage(remove_pers=True)
            print('save image!!!')
            cv2.imwrite('picture\\savedPic\\'+str(cam.name)+'test2.jpg',b1)
            plate, platePos, plateOrient8ation = IP.filter_plate(plate_img = plate, plate_position = platePos, plate_Orientation= plateOrientation)

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

        # cv2.imshow('test',selectPlate)
        # print(mainPredict([selectPlate],model=MODEL)[0])
        # cv2.waitKey(0)        
        if str(selectOreintation[0:3,2]) == '[ 0.  0. -1.]':
            wall = 'B'
        elif str(selectOreintation[0:3,2]) == '[-1.  0.  0.]':
            wall = 'L'
        elif str(selectOreintation[0:3,2]) == '[1. 0. 0.]':
            wall = 'R'

        data.append( [selectPosition, wall, mainPredict([selectPlate],model='CNN')[0] ,selectOreintation] )
    send_serial.getSetQAndWrite(POSITION_STEP_1,0)

for dat in data:
    print(data.index(dat)+1, ' ', dat)
# sleep(5)
input('data from detection :')

if len(data) > 10:
    data = data[:10]
if MODE_POSITION or TEST_MODE == False:
    # data = position wall predict orentation
        
    send_serial.getXYZAndWrite(data)
else :
    send_serial.getSetQAndWrite(data,0)
        
if TEST_MODE == False:
    send_serial.getSetQAndWrite(POSITION_STEP_1,0)
