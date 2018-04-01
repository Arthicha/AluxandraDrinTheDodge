__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'main program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

# 1. system module
import os
import sys
import copy

# 2. machine learning module
import tensorflow as tf

# 3. mathematical module
import numpy as np
import math
import random

# 4. our own module
from module.Tenzor import TenzorCNN,TenzorNN,TenzorAE
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
from module.Retinutella_theRobotEye import Retinutella

# 5. visualization module
import matplotlib.pyplot as plt

# 6. image processing module
import cv2

'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''

cam = Retinutella('cam1',0,0,cameraMode=1)

'''*************************************************
*                                                  *
*                   main program                   *
*                                                  *
*************************************************'''

# main loop
while(1):

    # get word from original picture
    # after pass through this section, we get list of word (image) in the variable
    # 'plate'.
    org,plate = cam.getListOfPlate()


    #show and finally destroy those windows.
    for i in range(0,len(plate)):
        cam.show(plate[i],frame='plate'+str(i))
    cam.show(org,wait=1)
    cam.destroyWindows()
