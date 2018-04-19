__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'main program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

# 1. system module
import os
import sys
import copy

# 2. machine learning module
import tensorflow as tf

# 3. mathematical module
import numpy as np
import math
import random

# 4. our own module
from module.Tenzor import TenzorCNN,TenzorNN,TenzorAE
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella
from module.RandomFunction import *
from module.Zkeleton import Zkele

# 5. visualization module
import matplotlib.pyplot as plt

# 6. image processing module
import cv2

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

# machine learning model CNN, KNN, RF and HAR
ML_CNN = 0
ML_KNN = 1
ML_RF = 2
ML_HAR = 3

PATH = os.getcwd()


'''*************************************************
*                                                  *
*               configuration variable             *
*                                                  *
*************************************************'''



IMAGE_SIZE = (32,64)
N_CLASS = 30

# select machine learning model
MODEL = ML_CNN

# restore save model
# for example, PATH+"\\savedModel\\modelCNN"
GETT_CNN_PATH = PATH+"\\savedModel\\modelCNN_edge"
GETT_KNN_PATH = PATH
GETT_RF_PATH = PATH
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



'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

cam = Retinutella('cam1',0,0,cameraMode=1)
cam2 = Retinutella('cam2',0,0,cameraMode=0)

NUM2WORD = ["0","1","2","3","4","5","6","7","8","9",
            "zero","one","two","three","four","five","six","seven","eight","nine",
            "soon","nung","song","sam","see","ha","hok","jed","pad","kaow"]

'''*************************************************
*                                                  *
*                 setup program                    *
*                                                  *
*************************************************'''

if MODEL not in range(0,3):
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
*                   main program                   *
*                                                  *
*************************************************'''

# main loop
while(1):

    # get word from original picture
    # after pass through this section, we get list of word (image) in the variable
    # 'plate'.
    corg = cam2.getImage()
    org,plate = cam.getListOfPlate(image_size=IMAGE_SIZE)


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
        if MODEL is ML_CNN:
            pred_result = sess.run(pred_prob,feed_dict={x: list_vector})



    #show and finally destroy those windows.
    for p in range(0,len(plate)):
        plate[p] = cv2.resize(plate[p],(IMAGE_SIZE[1]*5,IMAGE_SIZE[0]*5))
        cam.show(plate[p],frame='plate_'+str(NUM2WORD[pred_result[p]]))
        cv2.moveWindow('plate_'+str(NUM2WORD[pred_result[p]]), 700,300);
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(corg, str(NUM2WORD[pred_result[p]]), (50, 400), font, 5, (0, 0, 255), 5, cv2.LINE_AA)
    cam.show(corg,wait=30)
    cam.destroyWindows()
