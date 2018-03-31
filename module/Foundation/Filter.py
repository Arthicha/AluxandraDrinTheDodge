from scipy.signal import convolve2d
import cv2
import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
from skimage import restoration
from .Kuwahara import Kuwahara
import time

class Filter():
    def Filter_weiner(image,window_size=5,iteration=1):
        psf = np.ones((window_size, window_size)) / pow(window_size,2)
        image = sk.restoration.wiener(image, psf, iteration,clip=False)
        return image


    def Filter_median(image, window):
        image = cv2.medianBlur(image, window)
        return image


    def Filter_Kuwahara(image, window):
        image = Kuwahara(image, window)
        return image
