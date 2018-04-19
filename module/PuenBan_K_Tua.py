'''*************************************************
*                                                  *
*                  import module                   *
*                                                  *
*************************************************'''

import cv2
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, train_test_split,StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import matplotlib.pyplot as plt
import itertools
from module.commonKNN import clock, mosaic

dirSep = os.path.sep
class PuenBan_K_Tua:
    def __init__(self,path= os.getcwd() + dirSep+'savedModel'+dirSep+'modelKNN' ):
        self.path = path

    def confusionMat(self, correct_Labels, Predicted_Labels):
        # labels = ['0','1','2','3','4','5','6','7','8','9','zero','one','two','three','four','five','six','seven','eight','nine','ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH','SevenTH','EightTH','NineTH']
        labels = list([str(i) for i in range(30)])
        # print(labels)
        con_mat = confusion_matrix(correct_Labels, Predicted_Labels,labels=labels)
        # print(con_mat)
        # print(con_mat.shape)
        siz = con_mat.shape
        size = siz[0]
        total_pres = 0
        for i in range(size):
            total_pres = total_pres + (con_mat[i, i])
            print('Class accuracy '+str(i)+': '+str(con_mat[i, i] / float(np.sum(con_mat[i, :]))))
        print('total_accuracy : ' + str(total_pres/float(np.sum(con_mat))))
        df = pd.DataFrame (con_mat)
        filepath = self.path + os.path.sep + 'my_excel_file_PIC.xlsx'
        self.plot_confusion_matrix(con_mat, classes=labels,
                        title='Confusion matrix, without normalization')
        df.to_excel(filepath, index=False)
        plt.show()
    #correct_lables = matrix of true class of the test data
    #Predicted_labels = matrix of the predicted class

    def plot_confusion_matrix(self, cm, classes,
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


    def showPic(self, img):
        cv2.imshow("show",img)
        cv2.waitKey(0)
    #chg

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            # no deskewing needed.
            return img.copy()
        # Calculate skew based on central momemts.
        skew = m['mu11']/m['mu02']
        # Calculate affine transform to correct skewness.
        M = np.float32([[1, skew, -0.5**skew], [0, 1, 0]])
        # Apply affine transform
        img = cv2.warpAffine(img, M, (64, 32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img


    def HOG_int(self) :
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
        signedGradient = True
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
        return hog

    def plotKgraph(self, a, b):
        plt.plot(a, b)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('ACCURACY')
        plt.show()
