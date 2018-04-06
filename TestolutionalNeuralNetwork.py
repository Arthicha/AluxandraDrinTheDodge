__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'test program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

# 1. system module
import os
import sys
import copy
import itertools

# 2. machine learning module
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# 3. mathematical module
import numpy as np
import pandas as pd
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
FROM_IMAGE = 0

GETT_IMG_PATH = PATH + '\\dataset\\setPlate'#'\\dataset\\DatasetG2'

# select machine learning model
MODEL = ML_CNN

# restore save model
# for example, PATH+"\\savedModel\\modelCNN"
GETT_CNN_PATH = PATH+"\\savedModel\\modelCNN"
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def confusionMat(correct_Labels, Predicted_Labels):
    labels = range(0,30)#['0','1','2','3','4','5','6','7','8','9','zero','one','two','three','four','five','six','seven','eight','nine','ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH','SevenTH','EightTH','NineTH']
    plt.figure()
    np.set_printoptions(precision=2)
    con_mat = confusion_matrix(correct_Labels, Predicted_Labels,labels=labels)
    print(con_mat)
    print(con_mat.shape)
    siz = con_mat.shape
    size = siz[0]
    total_pres = 0
    for i in range(size):
        total_pres = total_pres + (con_mat[i, i])
        print('Class accuracy '+str(i)+': '+str(con_mat[i, i] / float(np.sum(con_mat[i, :]))))
    print('total_accuracy : ' + str(total_pres/float(np.sum(con_mat))))
    df = pd.DataFrame (con_mat)
    filepath = 'D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\my_excel_file_PIC.xlsx'
    plot_confusion_matrix(con_mat, classes=labels,
                      title='Confusion matrix, without normalization')
    df.to_excel(filepath, index=False)
    plt.show()

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
sum_accu = 0
n_data = 0
actualVSpredict = [[],[]]
if FROM_IMAGE:
    for num in range(0,10):
        print('process',num)
        for type in ['N','E','T']:
            for font in range(0,5):
                images = np.array(cv2.imread(GETT_IMG_PATH+'\\'+str(num)+type+str(font)+'.png',0))

                step =images.shape[1]//100
                for i in range(step,images.shape[1],step):
                    try:
                        img = images[:,i-step:i]
                        img = IP.Get_Word2([img],image_size=IMAGE_SIZE)[0]
                        #img = Zkele(img,method='3d')
                        img = IP.auto_canny(img,sigma=0.33)
                        img = 255-img
                        plate = np.array([img])
                        list_vector = np.resize(np.array(plate),(len(plate),IMAGE_SIZE[0]*IMAGE_SIZE[1]))
                        # convert 8 bit image to be in range of 0.00-1.00, dtype = float
                        list_vector = list_vector/255
                        clas = copy.copy(num)
                        if type == 'E':
                            clas += 10
                        elif type == 'T':
                            clas += 20
                        pred = sess.run(pred_prob,feed_dict={x: list_vector})
                        actualVSpredict[1].append(pred)
                        actualVSpredict[0].append(clas)
                        if pred == clas:
                            accu = 1.00
                        else:
                            accu = 0.00
                        sum_accu += accu
                        n_data += 1
                        print('accuracy',sum_accu/n_data,'from',n_data)
                        cv2.imshow('frame',img)
                        cv2.waitKey(1)

                    except:
                        pass
else:
    evalu = [0.0,0.0]
    for c in range(0,30):
        for f in range(0,10):
            image_name = str(c)+'_'+str(f)

            print(image_name)
            img = cv2.imread(GETT_IMG_PATH+'\\'+image_name+'.jpg',0)

            plate = IP.Get_Plate2(img,thres_kirnel=21,morph=True)

            word = IP.Get_Word2(plate,image_size=IMAGE_SIZE)
            for w in range(0,len(word)):
                word[w] = Zkele(word[w],method='3d')
            list_vector = np.resize(np.array(word),(len(plate),IMAGE_SIZE[0]*IMAGE_SIZE[1]))
            list_vector = list_vector/255
            clas = c
            pred = sess.run(pred_prob,feed_dict={x: list_vector})
            print(pred)
            for w in range(0,len(word)):
                if np.count_nonzero(np.array(word[w])) < 0.98*IMAGE_SIZE[0]*IMAGE_SIZE[1]:
                    if pred[w] == c:
                        evalu[0] += 1
                    evalu[1] += 1
            # loop through each word
            if 0:
                for w in range(0,len(word)):
                    cv2.imshow('word'+str(w),word[w])
            print(evalu[0]/evalu[1],'percent from', evalu[1],'image')
            cv2.imshow('org',img)
            cv2.waitKey(1)
confusionMat(actualVSpredict[0],actualVSpredict[1])




