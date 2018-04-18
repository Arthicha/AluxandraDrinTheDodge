import os
import numpy as np
from module.PuenBan_K_Tua import PuenBan_K_Tua
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from time import sleep
import cv2
'''*************************************************
*                                                  *
*                 configuration                    *
*                                                  *
*************************************************'''

testKNN = PuenBan_K_Tua()

# HOG parameters
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
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

# # model parameters
# test_hog_descriptors = []

# path = os.getcwd() +  dirSep + 'dataset' + dirSep + 'synthesis' + dirSep + 'textfile_skele'
# dirs = os.listdir(path)
# savepath = os.getcwd() + dirSep + 'savedModel' + dirSep + 'modelKNN'

# TEST_SC = []
# K = []

# data = np.random.sample((32,64)) *255


# imgs = data.astype(np.uint8)
# imgs = testKNN.deskew(imgs)
# test_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))

# model = joblib.load(savepath+ dirSep+ 'knn_model_real.pkl')
# print(dir(model))
# pred = model.predict(test_hog_descriptors)

# print(pred)

'''---------------------------------------'''

# model parameters
lables = []
test_hog_descriptors = []
test_lables = []
val_hog_descriptors = []
val_lables = []
path = os.getcwd() +  dirSep + 'dataset' + dirSep + 'synthesis' + dirSep + 'textfile_skele'
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
        imgs = imgs.reshape(-1,(60))
        
        num += 1
        imgs = testKNN.deskew(imgs)
        

        test_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))
        
# print(test_hog_descriptors)
test_hog_descriptors = np.squeeze(test_hog_descriptors)


model = joblib.load(savepath+ dirSep+ 'knn_model_real.pkl')

pred = model.predict(test_hog_descriptors.tolist())    
        
print(pred)