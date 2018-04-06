__author__ = ['Zumo Arthicha Srisuchinnawong']
__version__ = 2.0
__description__ = 'Data Manipulation Program'

import numpy as np
import os
import random
import copy

def getData(foldername,N_CLASS,IMG_SIZE,n=-1,readList=[0,1,2],ttv=[0,1,2],dtype=np.uint8):

    '''
    this function get dataset from compress file in
    data0-9compress folder,
    :parameter: foldername folder path
    :parameter: n = amount of data in each class
    :parameter: ttv = number of data to get in testing,
                training abd validating dataset
    :parameter: dtype is numpy data type np.uint8 or np.float32
    :return: testing, training and validating dataset
    '''


    TestTrainValidate = [[],[],[]]
    LabelTTT = [[],[],[]]

    suffix = ['test','train','validate']
    listOfClass = range(0,30)

    for s in readList:
        print('STATUS: process data',str(100.0*s/3.0))
        for j in range(0,N_CLASS):
            object = listOfClass[j]
            print('PROCESS:',foldername+'\\'+str(object)+'_'+suffix[s]+'.txt')
            f = open(foldername+'\\'+str(object)+'_'+suffix[s]+'.txt','r')

            image = str(f.read()).split('\n')[:n]
            f.close()
            for i in range(len(image)):
                image[i] = np.fromstring(image[i], dtype=dtype, sep=',')
                image[i] = np.array(image[i])/255.0
                image[i] = np.reshape(image[i],(IMG_SIZE[0]*IMG_SIZE[1]))
            TestTrainValidate[s] += image
            obj = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            obj[j] = 1
            LabelTTT[s] += np.full((len(image),N_CLASS),copy.deepcopy(obj)).tolist()
        if s == 1:
            if 1:
                print('STATUS: shuffle-ing')
                def shuffly(a,b):
                    c = list(zip(a, b))
                    random.shuffle(c)
                    a, b = zip(*c)
                    return a,b
                a,b = shuffly(TestTrainValidate[1],LabelTTT[1])
                trainingSet = [np.array(a),np.array(b)]
                print('STATIS: complete shuffle-ing')
        del image
        del object
    testingSet  = [TestTrainValidate[ttv[0]],LabelTTT[ttv[0]]]
    validationSet = [TestTrainValidate[ttv[2]],LabelTTT[ttv[2]]]
    return testingSet,trainingSet,validationSet
