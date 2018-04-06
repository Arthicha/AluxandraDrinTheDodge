# ~/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

from module.detectCascade import multiCascade


hc = multiCascade()

print('test cascade accuracy files.')
output = hc.testCascade(feature= 'HAAR')
print(hc.calculateAccuracy(output))
hc.displayConclusionMatrix(output)
        
