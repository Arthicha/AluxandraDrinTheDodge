# ~/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

from module.trainCascade import trainCascade

scaleWeightHeight = 0.5
scalePosNeg = 1
memoryUse = 4096
multiPos = 1

minHitRate = 0.960
maxFalseAlarmRate = 0.200
weightTrimRate = 0.65
maxDepth = 1
maxWeakCount = 80

hc = trainCascade()

hc.settingHyperParameter(scaleWeightHeight=scaleWeightHeight, scalePosNeg=scalePosNeg,
                        memoryUse=memoryUse, multiPos=multiPos, minHitRate=minHitRate,
                        maxFalseAlarmRate=maxFalseAlarmRate, weightTrimRate=weightTrimRate,
                        maxDepth=maxDepth, maxWeakCount=maxWeakCount)

hc.AutoGenerateClassification(numberPerClass=1400,numstate=2, size=16)

# hc.removeOldData()