__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 1.0
__description__ = 'Store Skeletoning Function'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

from skimage.morphology import skeletonize, skeletonize_3d
import cv2
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP


'''*************************************************
*                                                  *
*                 random function                  *
*                                                  *
*************************************************'''

def Zkele(image,sauvola=21,method='norm'):
    '''
    :param image: input image
    :param sauvola: size of sauvola kirnel
    :param method: method to find skeleton, method should be either normal ('norm') or 3d skeleton ('3d')
    :return: image of skeleton of the input image
    example
                    ske = Zkele(image,sauvola=3,method='3d')
        this function find the skeleton of input image,'image' by using Sauvola thresholding with kernel
    size equal to 3 and find skeleton by using 3d skeleton method.
    '''
    data = IP.binarize(image,method=IP.SAUVOLA_THRESHOLDING,value=sauvola)/255
    data = 1.00-data
    if method == 'norm':
        skeleton = skeletonize(data)
    elif method == '3d':
        skeleton = skeletonize_3d(data)
    skeleton = skeleton
    return 255.0 - skeleton