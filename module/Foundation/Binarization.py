import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib
import skimage.io as io
import skimage.filters as filt
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


class Binarization():
    def Binarization_Otsu(image):
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image

    def Binarization_LMM(image):
        contrast = image.astype(np.float64)
        Imax = 0
        Imin = 0
        n_pixel = len(image[1]) * len(image)
        for i in range(len(image)):
            for j in range(len(image[1])):
                neighbor = []
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if i + x >= 0 and i + x < len(image) and (j + y >= 0 and j + y < len(image[1])):
                            if x == 0 and y == 0:
                                pass
                            else:
                                neighbor.append(image[i + x][j + y])
                Imax = Imax + float(max(neighbor))
                Imin = Imin + float(min(neighbor))
                contrast[i][j] = float(int(max(neighbor)) - float(min(neighbor))) / (
                    float(int(max(neighbor)) + float(min(neighbor))) + np.finfo(float).eps)
        Thresh = ((Imax - Imin) / n_pixel) / (((Imax + Imin) / n_pixel) + np.finfo(float).eps)
        ret, image = cv2.threshold(contrast, Thresh, 255, cv2.THRESH_BINARY)

        image = 255 - image
        return image

    def Binarization_Niblack(image, window_size, k):
        thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
        binary_niblack = image > thresh_niblack
        binary_niblack = binary_niblack.astype(np.float32)
        binary_niblack = binary_niblack * 255
        binary_niblack = 255 - binary_niblack
        return binary_niblack

    def Binarization_Sauvola(image, window_size):
        thresh_sauvola = threshold_sauvola(image, window_size=window_size)
        binary_sauvola = image > thresh_sauvola
        binary_sauvola = binary_sauvola.astype(np.float32)
        binary_sauvola = binary_sauvola * 255
        return binary_sauvola

    def Binarization_Bernsen(image):
        contrast = image
        Imax = 0
        Imin = 0
        n_pixel = len(image[1]) * len(image)
        for i in range(len(image)):
            for j in range(len(image[1])):
                neighbor = []
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if i + x >= 0 and i + x < len(image) and (j + y >= 0 and j + y < len(image[1])):
                            if x == 0 and y == 0:
                                pass
                            else:
                                neighbor.append(image[i + x][j + y])
                Imax = Imax + int(max(neighbor))
                Imin = Imin + int(min(neighbor))
                contrast[i][j] = int(max(neighbor)) - int(min(neighbor))
        Thresh = (Imax - Imin) / n_pixel
        ret, image = cv2.threshold(contrast, Thresh, 255, cv2.THRESH_BINARY)
        return image

    def Binarization_Adapt(image, window_size, weight):
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, window_size, weight)
        return image
