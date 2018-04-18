# !#/home/kitti/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

from module.HaarLikeFeature import trainCascade

scaleWeightHeight = 0.5
scalePosNeg = 0.5
memoryUse = 4096
multiPos = 1

minHitRate = 0.980
maxFalseAlarmRate = 0.08
weightTrimRate = 0.95
maxDepth = 1
maxWeakCount = 80

hc = trainCascade()

hc.settingHyperParameter(scaleWeightHeight=scaleWeightHeight, scalePosNeg=scalePosNeg,
                        memoryUse=memoryUse, multiPos=multiPos, minHitRate=minHitRate,
                        maxFalseAlarmRate=maxFalseAlarmRate, weightTrimRate=weightTrimRate,
                        maxDepth=maxDepth, maxWeakCount=maxWeakCount)

hc.AutoGenerateClassification(numberPerClass=0,numstate=1, size=16)
# hc.copyUsedModel()
# hc.removeOldData()
