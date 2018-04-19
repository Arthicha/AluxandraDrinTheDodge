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
from sklearn.externals import joblib
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
from module.ManipulateTaDa import getData
from module.PuenBan_K_Tua import PuenBan_K_Tua
from module import PaZum

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



FROM_LIST = 0
FROM_IMAGE = 1
FROM_WORD = 2
FROM_COMPRESS = 3

MODE = FROM_COMPRESS
FONTSHIFT = 1
NUM_IMG = 100

GETT_IMG_PATH = PATH + '\\dataset\\DatasetG4'#'\\dataset\\DatasetG2'
GETT_COMPRESS_PATH = PATH + '\\dataset\\synthesis\\textfile_skele'
# select machine learning model
MODEL = ML_CNN
sep = os.path.sep

# restore save model
# for example, PATH+"\\savedModel\\modelCNN"
GETT_CNN_PATH = PATH+"\\savedModel\\modelCNN_skele"
GETT_KNN_PATH = PATH+sep+'savedModel'+sep+'modelKNN'
GETT_RF_PATH = PATH+"\\savedModel\\modelRandomForest\\Random_Forest_best_run.pkl"
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

    #print(cm)

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
    #print(con_mat)
    #print(con_mat.shape)
    siz = con_mat.shape
    size = siz[0]
    total_pres = 0

    sumVerHor = np.zeros((2,30))
    Precision = np.zeros(30)
    Recall = np.zeros(30)
    F1 = np.zeros(30)
    for i in range(size):
        for j in range(size):
            sumVerHor[0][i] += con_mat[j,i]
            sumVerHor[1][i] += con_mat[i,j]
        if con_mat[i,i] != 0:
            Precision[i] = con_mat[i,i]/sumVerHor[0,i]
            Recall[i] = con_mat[i,i]/sumVerHor[1,i]
            F1[i] = (Recall[i] * Precision[i] * 2.00) / (Recall[i] + Precision[i])
        else:
            Precision[i] = 0.00
            Recall[i] = 0.00
            F1[i] = 0.00
    for i in range(size):
        total_pres = total_pres + (con_mat[i, i])
        print('---------------------------------------------------')
        print('Class',i)
        print('\t\taccuracy '+': '+str(con_mat[i, i] / float(np.sum(con_mat[i, :]))))
        print('\t\tprecision ' + ': ' + str(Precision[i]))
        print('\t\trecall ' + ': ' + str(Recall[i]))
        print('\t\tf1-score ' + ': ' + str(F1[i]))

    print('total_accuracy : ' + str(total_pres/float(np.sum(con_mat))))
    print('average_precision :',np.sum(Precision)/size)
    print('average_recall :',np.sum(Recall)/size)
    print('average_f1-score',np.sum(F1)/size)
    df = pd.DataFrame (con_mat)
    filepath = 'D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\my_excel_file_PIC.xlsx'
    plot_confusion_matrix(con_mat, classes=labels,
                      title='Confusion matrix, without normalization')
    df.to_excel(filepath, index=False)
    plt.show()


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
sum_accu = 0
n_data = 0
actualVSpredict = [[],[]]
if MODE == FROM_LIST:
    for num in range(0,10):
        print('process',num)
        for type in ['N','E','T']:
            for font in range(0+FONTSHIFT,5+FONTSHIFT):

                images = np.array(cv2.imread(GETT_IMG_PATH+'\\'+str(num)+type+str(font)+'.png',0))

                step =100
                img_count = 0
                for i in range(step,images.shape[1],step):
                    if images.shape[0] > step:
                        step_row = 100
                    else:
                        step_row = 0
                    for j in range(0,images.shape[0]-step_row,step):
                        try:
                            if step_row == 0:
                                img = images[:, i - step:i]
                            else:
                                img = images[j:j+step,i-step:i]
                            img = IP.Get_Word2([img],image_size=IMAGE_SIZE)[0]
                            img = Zkele(img,method='3d')
                            #img = IP.auto_canny(img,sigma=0.33)
                            #img = 255-img
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
                            img_count += 1
                            if img_count > NUM_IMG:
                                break
                        except:
                            pass
                    if img_count > NUM_IMG:
                        break
elif MODE == FROM_IMAGE:
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
elif MODE == FROM_COMPRESS:
    true_label = []
    actualVSpredict = [[],[]]
    sett = 0
    for i in range(0,300,50):
        print('testing section',i)
        testingset,trainingset,validationset = getData(GETT_COMPRESS_PATH,30,IMAGE_SIZE,ni=i,n=i+50,readList=[sett,sett,sett],ttv=[sett,sett,sett])
        # pred = sess.run(y_pred,feed_dict={x: testingset[0]})
        pred_result1 = sess.run(y_pred, feed_dict={x: testingset[0]})
        pred_result2 = predictKNN(testingset[0] * 255)
        pred_result3 = forest.predict(testingset[0], 'prob')
        pred = log_prob(pred_result1, pred_result2, pred_result3)
        true_label = np.argmax(testingset[1],axis=1)
        actualVSpredict[0] += list(pred)
        actualVSpredict[1] += list(true_label)
confusionMat(actualVSpredict[0],actualVSpredict[1])




