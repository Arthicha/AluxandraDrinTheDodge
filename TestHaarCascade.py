# ~/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

from module import FAI
from module.HaarLikeFeature import HaarLikeFeature


hc = HaarLikeFeature()

# print('test cascade accuracy files.')

# hc.settingHyperParameter(scaleFactor=1.2, minNeighbors= 3)
# output = hc.testCascade()
# # print(output)
# hc.displayConfusionMatrix(output)
data = [str(i)+'_test' for i in list(range(30))]
# output = hc.predict(data= data,Return='class')
# print(output)

# print(hc.calculateAccuracy(output))
# hc.displayConfusionMatrix(output)
     
print(FAI.PReF1(model='HAAR',test_data=data))
