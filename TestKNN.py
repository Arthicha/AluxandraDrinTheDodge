import os
import numpy as np
from module.PuenBan_K_Tua import PuenBan_K_Tua
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from time import sleep
import cv2
import copy
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella
from module.RandomFunction import *
from module.Zkeleton import Zkele

import matplotlib.pyplot as plt
'''*************************************************
*                                                  *
*                 configuration                    *
*                                                  *
*************************************************'''

testCode = 1

testKNN = PuenBan_K_Tua()

# HOG parameters
winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
hog = testKNN.HOG_int()
hog_descriptors = []

dirSep = os.path.sep

'''---------------------------------------'''

# model parameters
def detect(img=[]):
    test_hog_descriptors = []
    rePred = []
    savepath = os.getcwd() + dirSep + 'savedModel' + dirSep + 'modelKNN'

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
            model = joblib.load(savepath+ dirSep+ 'knn_model_real.pkl','r')

            pred = model.predict(test_hog_descriptors.tolist())    
            rePred.append(pred[0])
        except:
            rePred.append(pred[0])
    return rePred

'''---------------------------------------'''
if testCode ==0:
    # model parameters
    lables = []
    test_hog_descriptors = []
    test_lables = []
    val_hog_descriptors = []
    val_lables = []
    # path = os.getcwd() +  dirSep + 'dataset' + dirSep + 'synthesis' + dirSep + 'textfile_skele'
    path = os.getcwd() +  dirSep + 'dataset' + dirSep + 'real'
    dirs = os.listdir(path)
    savepath = os.getcwd() + dirSep + 'savedModel' + dirSep + 'modelKNN'

    TEST_SC = []
    K = []

    for files in dirs:
        
        directors = open(path+dirSep+str(files),'r')
        datas = directors.read()
        directors.close()
        datas=datas.split('\n')
        datas=datas[:-1]
        num =0
        
        for z in datas:
            
            lisss=z.split(',')
            imgs = np.array(list(lisss[:]))

            imgs = imgs.reshape(-1,(64))
            imgs = imgs.astype(np.uint8)
            imgs = testKNN.deskew(imgs)
            
            test_hog_descriptors.append(hog.compute(imgs,winStride=(16,16)))
            break    
    test_hog_descriptors = np.squeeze(test_hog_descriptors)


    model = joblib.load(savepath+ dirSep+ 'knn_model_real.pkl','r')
    pred = model.predict(test_hog_descriptors.tolist())    
            
    print(pred)
    

'''---------------------------------------'''

if testCode == 1:
    IMAGE_SIZE = (32,64)
    cam = Retinutella('cam1',1,0,cameraMode=1)
    cam2 = Retinutella('cam2',1,0,cameraMode=0)
    NUM2WORD = ["0","1","2","3","4","5","6","7","8","9",
                "zero","one","two","three","four","five","six","seven","eight","nine",
                "soon","nung","song","sam","see","ha","hok","jed","pad","kaow"]
    while(1):
        
        # get word from original picture
        # after pass through this section, we get list of word (image) in the variable
        # 'plate'.

        corg = cam.getImage()
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
            # print(plate)
            list_vector = np.resize(np.array(plate),(len(plate),IMAGE_SIZE[0]*IMAGE_SIZE[1]))
            
            # convert 8 bit image to be in range of 0.00-1.00, dtype = float
            # list_vector = list_vector/255
            
            outp = detect(list_vector)

        #show and finally destroy those windows.
        for p in range(0,len(plate)):
            plate[p] = cv2.resize(plate[p],(IMAGE_SIZE[1]*5,IMAGE_SIZE[0]*5))
            cam.show(plate[p],frame='plate_'+str(NUM2WORD[int(outp[p])]))
            
            cv2.moveWindow('plate_'+str(NUM2WORD[int(outp[p])]), 700,300);
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(corg, str(NUM2WORD[int(outp[p])]), (50, 400), font, 5, (0, 0, 255), 5, cv2.LINE_AA)
        cam.show(corg,wait=30)
        cam.destroyWindows()
