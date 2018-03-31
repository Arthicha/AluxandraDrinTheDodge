__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 1.0
__description__ = 'Store Random Function'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import random
import sys

import cv2
import numpy as np

from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr

'''*************************************************
*                                                  *
*                 random function                  *
*                                                  *
*************************************************'''

def RND_MAGNIFY(img,MAGNIFY = [90,110],step = 5):

    '''
    :param img: input image
    :param MAGNIFY: list of magnification range [x,y] from x to y (percent)
    :param step: step of magnification
    :return: magnified image
    example
            mag_img = RND_MAGNIFY(img,MAGNIFY=[50,150],step=25)
        this function take img as an input image, then random the magnification value in [50, 75, 100, 125,
    150] (from 50-150, step 25)
    '''
    if len(img.shape) != 2:
        sys.exit('image size error in RND_MAGNIFY function')
    if len(MAGNIFY) != 2:
        sys.exit('magnification range size error in RND_MAGNIFY function')
    if (type(MAGNIFY[0]) != int) or (type(MAGNIFY[1]) != int):
        sys.exit('magnification range dont have data type \'interger\' in RND_MAGNIFY function')
    if type(step) != int:
        sys.exit('magnification step dont have data type \'interger\' in RND_MAGNIFY function')
    magnify_value = np.arange(MAGNIFY[0],MAGNIFY[1],step).tolist()
    magnify_img = ipaddr.magnifly(img, percentage=magnify_value[random.randint(0,len(magnify_value)-1)])
    return magnify_img

def RND_MORPH(img,MORPH = [1,5],step = 1):

    '''
    :param img: input image
    :param MORPH: list of morphological transformation range [x,y] from x to y
    :param step: step of morphological transformation
    :return: morphed image
    example
            morp_img = RND_MORPH(img,MORPH=[1,5],step=1)
        this function take img as an input image, then random the morphological transformation value in
    [1, 2, 3, 4, 5] (from 1-5, step 1)
    '''
    if len(img.shape) != 2:
        sys.exit('image size error in RND_MORPH function')
    if len(MORPH) != 2:
        sys.exit('morphological transformation range size error in RND_MORPH function')
    if (type(MORPH[0]) != int) or (type(MORPH[1]) != int):
        sys.exit('morphological transformation range dont have data type \'interger\' in RND_MORPH function')
    if type(step) != int:
        sys.exit('morphological transformation step dont have data type \'interger\' in RND_MORPH function')
    morph_value = np.arange(MORPH[0],MORPH[1],step).tolist()
    MODE = ipaddr.DILATE
    if random.randint(0,1):
        MODE = ipaddr.ERODE
    value = random.choice(morph_value)
    # morph_value[random.randint(0,len(morph_value)-1)],morph_value[random.randint(0,len(morph_value)-1)]
    morph_image =ipaddr.morph(img,MODE,value=[value,value])
    return morph_image

def RND_MOVE(img,MOVE=[-3,3],step=2):

    '''
    :param img: input image
    :param MORPH: list of translation range [x,y] from x to y
    :param step: step of translation transformation
    :return: translate image
    example
            mov_img = RND_MOVE(img,MOVE=[-3,3],step=2)
        this function take img as an input image, then random the translation value in [-3, -1, 1, 3]
    (from -3-3, step 2)
    '''
    if len(img.shape) != 2:
        sys.exit('image size error in RND_MOVE function')
    if len(MOVE) != 2:
        sys.exit('translation range size error in RND_MOVE function')
    if (type(MOVE[0]) != int) or (type(MOVE[1]) != int):
        sys.exit('translation range dont have data type \'interger\' in RND_MOVE function')
    if type(step) != int:
        sys.exit('translation step dont have data type \'interger\' in RND_MOVE function')
    movex_value = np.arange(MOVE[0],MOVE[1],step).tolist()
    movey_value = np.arange(MOVE[0],MOVE[1],step).tolist()
    valuex = random.choice(movex_value)
    valuey = random.choice(movey_value)
    mov_image = ipaddr.translate(img, (valuex, valuey),[cv2.INTER_LINEAR, ipaddr.BORDER_CONSTANT, 255])
    return mov_image

def RND_GAMMA(image,GAMMA = [10,40],step=1):

    '''
    :param image: input image
    :param GAMMA: list of gamma transformation range [x,y] from x to y (10 x gamma)
    :param step: step of gamma transformation
    :return: translate image
    example
            gam_img = RND_GAMMA(img,GAMMA=[-3,3],step=2)
        this function take img as an input image, then random the gamma transformation value in [11, 12,..., 40]
    (from 10-40, step 1)
    '''

    if len(image.shape) != 2:
        sys.exit('image size error in RND_GAMMA function')
    if len(GAMMA) != 2:
        sys.exit('gamma range size error in RND_GAMMA function')
    if (type(GAMMA[0]) != int) or (type(GAMMA[1]) != int):
        sys.exit('gamma range don\'t have data type \'interger\' in RND_GAMMA function')
    if type(step) != int:
        sys.exit('gamma step don\'t have data type \'interger\' in RND_GAMMA function')

    value = np.arange(GAMMA[0],GAMMA[1],step).tolist()
    gamma = float(random.choice(value))/10.0
    if random.randint(0,1):
        gamma = 1.0/gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
