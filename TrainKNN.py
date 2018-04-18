
import os
import numpy as np
from module.PuenBan_K_Tua import PuenBan_K_Tua
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
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

# model parameters
lables = []
test_hog_descriptors = []
test_lables = []
val_hog_descriptors = []
val_lables = []
path = os.getcwd() +  dirSep + 'dataset' + dirSep + 'synthesis' + dirSep + 'textfile_skele'
dirs = os.listdir(path)
savepath = os.getcwd() + dirSep + 'savedModel' + dirSep + 'modelKNN'

TEST_SC = []
K = []



'''*************************************************
*                                                  *
*                 main function                    *
*                                                  *
*************************************************'''

#Import test and training data
for files in dirs:
    a = files.split('_')
    d = a[len(a)-1]
    if d == 'train.txt':
        lab = a[len(a)-2]
        director = open(path+dirSep+str(files),'r')
        data = director.read()
        director.close()
        data=data.split('\n')
        data=data[:-1]
        num =0
        for x in data:
            lisss=x.split(',')
            img = np.array(list(lisss[:]))
            img = img.reshape(-1,(64))
            img = img.astype(np.uint8)
            num += 1
            img = testKNN.deskew(img)
            hog_descriptors.append(hog.compute(img,winStride=(20,20)))
            lables.append(str(lab))
        # print('appended train '+str(files))
    if d == 'test.txt':
        labs = a[len(a)-2]
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
            num += 1
            imgs = testKNN.deskew(imgs)
            test_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))
            test_lables.append(str(labs))
        # print('appended test '+str(files))
    if d == 'validate.txt':
        labs = a[len(a)-2]
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
            num += 1
            imgs = testKNN.deskew(imgs)
            val_hog_descriptors.append(hog.compute(imgs,winStride=(20,20)))
            val_lables.append(str(labs))
        # print('appended test '+str(files))
hog_descriptors = np.squeeze(hog_descriptors)
lables = np.squeeze(lables)

test_hog_descriptors = np.squeeze(test_hog_descriptors)
test_lables = np.squeeze(test_lables)

val_hog_descriptors = np.squeeze(val_hog_descriptors)
val_lables = np.squeeze(val_lables)

print('Begining feature selection...')
#feature selection
forest = ExtraTreesClassifier()
forest.fit(hog_descriptors, lables)
modeltree = SelectFromModel(forest,prefit=True)
X_new = modeltree.transform(hog_descriptors)
test_new = modeltree.transform(test_hog_descriptors)

#change the range of m to find k parameter
for m in range(8,9):
    print('REAL DATA FILE TRAINING********************************************* K :'+str(m))
    # print('Begining Knn fitting...')
    neigh = KNeighborsClassifier(n_neighbors=m)
    neigh.fit(X_new, lables)
    # save model
    joblib.dump(neigh, savepath+ dirSep+ 'knn_model_real.pkl')
    print('Model saved!')
    pred = neigh.predict(test_new)
    print('predicted....')
    best_score=0
    all_hog_descriptors= [val_hog_descriptors,hog_descriptors,test_hog_descriptors]
    all_target =[val_lables,lables,test_lables]
    for i in range(0,2):
            train_feature = all_hog_descriptors[i].tolist()+all_hog_descriptors[i+1].tolist()
            train_target =  all_target[i].tolist()+all_target[i+1].tolist()
            neigh.fit(train_feature, train_target)
            train_score = neigh.score(train_feature, train_target)
            # print('test no :'+str(m)+' loop no : ' + str(i))
            # print("train score :   " + str(train_score))
            test_score = neigh.score(all_hog_descriptors[(i+2)%3].tolist(), all_target[(i+2)%3].tolist())
            # print("test score :   " + str(test_score))
            Label_Pred = neigh.predict(all_hog_descriptors[(i+2)%3].tolist())
            # confusionMat(all_target[(i+2)%3].tolist(), Label_Pred)
            # print(classification_report( all_target[(i+2)%3].tolist(), Label_Pred))
            if test_score > best_score:
                best_score = test_score
                best_test_target =all_target[(i+2)%3].tolist()
                best_pred=Label_Pred
                s = neigh
    testKNN.confusionMat(best_test_target, best_pred)
    print(best_score)
    TEST_SC.append(best_score)
    K.append(m)
    print('*************************************************************************************')
testKNN.plotKgraph(K, TEST_SC)

