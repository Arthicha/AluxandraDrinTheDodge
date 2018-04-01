__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'train and save convolutional neural network model'

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
from module.ManipulateTaDa import getData

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

# learning algorithm gradient descent of ADAM optimiser
LA_GRAD = 0
LA_ADAM = 1

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
GETT_PATH = None#PATH+"\\savedModel\\modelCNN"

SAVE_PATH = PATH+'\\savedModel\\modelCNN'

CONTINUE = False
AUGMENT = AUG_NONE
AUG_VALUE = [20,3]
MAGNIFY = [90,110]
MORPH = [1,5]
MOVE = [-3,3]

# convolutional neural network config
CNN_HIDDEN_LAYER = [32,64,128]
KERNEL_SIZE = [[5,5],[3,3]]
POOL_SIZE = [[4,4],[2,2]]
STRIDE_SIZE = [4,2]
KEEP_PROB = 0.9

# learning algorithm config
LEARNING_ALGORITHM = LA_ADAM
LEARNING_RATE = 0.01
BATCH2PRINT = 20
BATCH_SIZE = 2000
EPOCH = 2000
DIVIDEY = 1
VALIDATE_SECTION = 20

# set numpy to print/show all every element in matrix
np.set_printoptions(threshold=np.inf)

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

'''*************************************************
*                                                  *
*                     function                     *
*                                                  *
*************************************************'''
def accuracyPlot(y,graphName,xLabel,yLabel,saveAs):
    global SAVE_PATH
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(y)
    plt.title(graphName)
    plt.ylabel(xLabel)
    plt.xlabel(yLabel)
    ax.legend()
    fig.savefig(SAVE_PATH+'\\'+saveAs+'.png')


'''*************************************************
*                                                  *
*                   main program                   *
*                                                  *
*************************************************'''
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# read data from compress text file
test,train,validate = getData(PATH+'\\dataset\\synthesis\\textfile',N_CLASS,IMAGE_SIZE,n=-1,readList=[1],ttv=[1,1,1])

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
    keep_prob = tf.placeholder(tf.float32)

# create model of convolutional neural network
with tf.name_scope('CNN_model'):
    CNN = TenzorCNN()
    y_pred,activity = CNN.CNN2(x_image,CNN_HIDDEN_LAYER,KERNEL_SIZE,POOL_SIZE,STRIDE_SIZE,IMAGE_SIZE)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))

with tf.name_scope('gradient_descent_learning_algorithm'):

    if LEARNING_ALGORITHM is 'GRAD':
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    elif LEARNING_ALGORITHM is 'ADAM':
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.name_scope('evaluation'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if GETT_PATH != None:
    saver.restore(sess, GETT_PATH+'\\modelCNN.ckpt')
    print("Get model from path: %s" % GETT_PATH+'\\modelCNN.ckpt')

epoch_acc = []

for epoch in range(EPOCH):
    for i in range(BATCH_SIZE,len(train[1])//DIVIDEY,BATCH_SIZE):
        batch = [train[0][i-BATCH_SIZE:i],train[1][i-BATCH_SIZE:i]]

        if (i//BATCH_SIZE) % BATCH2PRINT == 0:
            train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            validation_accuracy = 0
            n_sec = 0
            for k in range(0,len(validate[1]),len(validate[1])//VALIDATE_SECTION):
                validation_accuracy += sess.run(accuracy,feed_dict={x: validate[0][k:k+len(validate[1])//VALIDATE_SECTION], y_: validate[1][k:k+len(validate[1])//VALIDATE_SECTION], keep_prob: 1.0})
                n_sec += 1
            validation_accuracy = validation_accuracy/n_sec
            print('EPOCH %d: step %d, training accuracy %g, validation accuracy %g' % (epoch,i, train_accuracy,validation_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})

    testing_accuracy = 0
    n_sec = 0
    for i in range(0,len(test[1]),len(test[1])//VALIDATE_SECTION):
        testing_accuracy += sess.run(accuracy,feed_dict={x: test[0][i:i+len(test[1])//VALIDATE_SECTION], y_: test[1][i:i+len(test[1])//VALIDATE_SECTION], keep_prob: 1.0})
        n_sec += 1
    testing_accuracy = testing_accuracy/n_sec
    print('EPOCH %d: test accuracy %g' % (epoch,testing_accuracy))

    validation_accuracy = 0
    n_sec = 0
    for k in range(0,len(validate[1]),len(validate[1])//VALIDATE_SECTION):
        validation_accuracy += sess.run(accuracy,feed_dict={x: validate[0][k:k+len(validate[1])//VALIDATE_SECTION], y_: validate[1][k:k+len(validate[1])//VALIDATE_SECTION], keep_prob: 1.0})
        n_sec += 1
    validation_accuracy = validation_accuracy/n_sec
    print('model accuracy:', validation_accuracy)
    if (SAVE_PATH != None): #and (validation_accuracy >= best_accuracy):

        best_accuracy = validation_accuracy
        filename = 'CNN'

        epoch_acc.append(float(validation_accuracy))
        accuracyPlot(epoch_acc,'accuracy of the model in each epoch','accuracy (prob)','time (epoch)','accuracy'+str(filename))
        f = open(SAVE_PATH+'\\model_'+filename+'_config.txt','w')
        f.write(str(float(validation_accuracy))+'\n')


        for each_layer in CNN_HIDDEN_LAYER:
            f.write(str(each_layer)+',')
        f.write('\n')
        f.write(str(BATCH_SIZE)+'\n')
        f.write(str(KEEP_PROB)+'\n')
        f.write(str(LEARNING_RATE)+'\n')
        f.close()
        save_path = saver.save(sess, SAVE_PATH+'\\modelCNN.ckpt')
        print("Model saved in path: %s" % save_path)



