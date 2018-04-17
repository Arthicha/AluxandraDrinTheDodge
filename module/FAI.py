
import os, shutil
import sys
 
import numpy as np



def tune(model='', Range=[]):
    
    keepValue = []

    return keepValue

def PReF1(model='', test_data=[]):
    
    from module.PuenBan_K_Tua import PuenBan_K_Tua
    from module.HaarLikeFeature import HaarLikeFeature

    keepValue = np.zeros((31,4))
    if model == 'KNN':
        pass
    elif model == 'RF':
        pass
    elif model == 'HAAR':
        hc = HaarLikeFeature()
        hc.settingHyperParameter(scaleFactor=1.2, minNeighbors= 3)
        value = hc.testCascade()
    
    keepFN = 0
    keepTP = 0
    keepFP = 0
    keepTN = 0   

    for row in range(30):
        FN = 0
        TP = 0
        FP = 0
        TN = 0
        for col in range(30):
            if row == col:
                TP += value[row][col]
                keepTP += value[row][col]

            elif row != col:
                FP += value[row][col]
                keepFP += value[row][col]
        
        FN = sum(value[row])-TP
        # TN = sum(value[row])-TP-FN-FP

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
        accuracy = TP/np.sum(value[row])
        keepValue[row] = [precision, recall, accuracy, F1]

    precision = keepTP/(keepTP+keepFP)
    recall = keepTP/(keepTP+keepFN)
    F1 = 2*precision*recall/(precision+recall)
    accuracy = keepTP/np.sum(value)

    keepValue[-1] = [precision, recall, accuracy, F1]

    return keepValue


def logAvg(prediction=[] ):
    
    n,m = prediction.shape
    keepValue = np.zeros((m,30))

    for col in range(m):
        for row in range(n):
            if type(prediction[row, col]) == int or type(prediction[row, col]) == np.int_:
                keepValue[col, prediction[row, col]] += 1

            else:
                for val in prediction[row, col]:
                    keepValue[col, val] += 1

    return keepValue

def prob2Class(probs = np.zeros((1,30))):
    
    keepValue = np.zeros((30,1))
    maxClass = []

    for selectClass in range(30):

        if float(max(keepValue)) < sum(probs[:,selectClass]):
            maxClass = [selectClass]

        elif float(max(keepValue)) == sum(probs[:,selectClass]):
            maxClass.append(selectClass)

        keepValue[selectClass] = sum(probs[:,selectClass])
    
    if len(maxClass) == 1:
        maxClass = maxClass[0]

    return maxClass

