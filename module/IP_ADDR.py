__author__ = ['Zumo', 'Tew', 'Wisa']
__version__ = 2.0

import sys
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from module.Foundation import Binarization, Filter
import copy


class Image_Processing_And_Do_something_to_make_Dataset_be_Ready():
    # def __init__(self):
    # GLOBAL
    SHAPE = (640, 480)
    CREATE_SHAPE = (320, 320)

    # Binarization Mode
    OTSU_THRESHOLDING = 0
    ADAPTIVE_CONTRAST_THRESHOLDING = 1
    NIBLACK_THRESHOLDING = 2
    SAUVOLA_THRESHOLDING = 3
    BERNSEN_THRESHOLDING = 4
    LMM_THRESHOLDING = 5

    # Noise Filter
    KUWAHARA = 6
    WIENER = 7
    MEDIAN = 8
    # Blur Filter
    GAUSSIAN = 9
    AVERAGING = 10
    BILATERAL = 11

    # morphology method
    ERODE = cv2.MORPH_ERODE
    DILATE = cv2.MORPH_DILATE
    OPENING = cv2.MORPH_OPEN
    CLOSING = cv2.MORPH_CLOSE
    GRADIENT = cv2.MORPH_GRADIENT
    TOP_HAT = cv2.MORPH_TOPHAT
    BLACK_HAT = cv2.MORPH_BLACKHAT

    # Estimate method
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_AREA = cv2.INTER_AREA
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    INTER_MAX = cv2.INTER_MAX
    WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP

    # Generate Border Method
    BORDER_CONSTANT = cv2.BORDER_CONSTANT
    BORDER_REPLICATE = cv2.BORDER_REPLICATE
    BORDER_REFLECT = cv2.BORDER_REFLECT
    BORDER_WRAP = cv2.BORDER_WRAP
    BORDER_REFLECT101 = cv2.BORDER_REFLECT101
    BORDER_TRANSPARENT = cv2.BORDER_TRANSPARENT
    BORDER_ISOLATED = cv2.BORDER_ISOLATED

    # Colorcode
    GRAY_SCALE = cv2.COLOR_BGR2GRAY
    RGB = cv2.COLOR_BGR2RGB
    HSV = cv2.COLOR_BGR2HSV
    BGR = None

    # pass


    def binarize(image, method=OTSU_THRESHOLDING, value=None):
        if method == __class__.OTSU_THRESHOLDING:
            img = Binarization.Binarization_Otsu(image)
        elif method == __class__.ADAPTIVE_CONTRAST_THRESHOLDING:
            img = Binarization.Binarization_Adapt(image, value[0], value[1])
        elif method == __class__.NIBLACK_THRESHOLDING:
            img = Binarization.Binarization_Niblack(image, value[0], value[1])
        elif method == __class__.SAUVOLA_THRESHOLDING:
            img = Binarization.Binarization_Sauvola(image, value)
        elif method == __class__.BERNSEN_THRESHOLDING:
            img = Binarization.Binarization_Bernsen(image)
        elif method == __class__.LMM_THRESHOLDING:
            img = Binarization.Binarization_LMM(image)
        else:
            sys.exit("Unknown method\n")
        return img.astype(np.uint8)
    # Binarize image into two value 255 or 0
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.binarize(img,method=ipaddr.OTSU_THRESHOLDING)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def remove_noise(image, method=KUWAHARA, value=5):
        if method == __class__.KUWAHARA:
            img = Filter.Filter_Kuwahara(image, value)
        elif method == __class__.WIENER:
            img = Filter.Filter_weiner(image, window_size=value[0], iteration=value[1])
        elif method == __class__.MEDIAN:
            img = Filter.Filter_median(image, window=value)
        else:
            sys.exit("Unknown method\n")
        return img
    # reduce image noise such as salt and pepper noise
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.remove_noise(img,method=ipaddr.KUWAHARA , value=5)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def resize(image, shape=SHAPE, method=INTER_LINEAR):
        img = cv2.resize(image, shape, interpolation=method)
        return img
    # change image size
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.resize(img,(28,28) )
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def capture(cam, size=SHAPE, mode=GRAY_SCALE, config=None):
        ret, image = cam.read()
        if mode != __class__.BGR:
            image = cv2.cvtColor(image, mode, config)
        image = cv2.resize(image, size)
        return image
    # change image size
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       camera = cv2.videoCapture(0)
       img = ipaddr.capture( camera)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''
    def translate(image, value, config=None):
        matrix = np.float32([[1, 0, value[0]], [0, 1, value[1]]])
        if config is None:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]))
        elif config[1] == __class__.BORDER_CONSTANT:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]), flags=config[0],
                                 borderMode=config[1], borderValue=config[2])
        else:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]), flags=config[0],
                                 borderMode=config[1])

        return img
    # translate image (move to the left right or whatever by value)
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.translate(img,(1,1))
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def blur(image, method=AVERAGING, value=5):
        if method == __class__.MEDIAN:
            img = cv2.medianBlur(image, value)
        elif method == __class__.AVERAGING:
            averaging_kernel = np.ones([value, value], dtype=np.float32) / pow(value, 2)
            img = cv2.filter2D(image, -1, averaging_kernel)
        elif method == __class__.GAUSSIAN:
            img = cv2.GaussianBlur(image, (value, value), 0)
        elif method == __class__.BILATERAL:
            img = cv2.bilateralFilter(image, value[0], value[1], value[2])
        else:
            sys.exit("Unknown method\n")
        return img
    # blur image with three method
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.blur(img,ipaddr.AVERAGING,3)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def morph(image, mode=DILATE, value=[3, 3]):
        matrix = np.ones((value[0], value[1]), np.float32)
        img = cv2.morphologyEx(image, mode, matrix)
        return img
    # morph image acording to mode and value use to construct kernel
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''



    def rotation(image, center_of_rotation, angle):
        matrix = cv2.getRotationMatrix2D((center_of_rotation[0], center_of_rotation[1]), angle, 1)
        # print(matrix)
        # print(image.shape)
        img = cv2.warpAffine(image, matrix, (image.shape[1],image.shape[0]), borderMode=__class__.BORDER_CONSTANT,
                             borderValue=255)
        # print(img.shape)
        return img
    # rotate image according to center of rotation and angle
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.rotation(img,(img.shape[1]/2,img.shape[2]/2),15)
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def font_to_image(font, size=CREATE_SHAPE, index=0, string="0"):
        # Create Plate from font and word
        # Example
        # from IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
        # image = ipaddr.font_to_image("angsana.ttc", 10, 0, "หนึ่ง")
        # cv2.imshow("one", image)
        # cv2.waitKey(0)

        Text_Font = ImageFont.truetype(font, size, index, encoding="unic")
        w, h = Text_Font.getsize(string)
        img = Image.new("L", __class__.CREATE_SHAPE, color=255)
        image = ImageDraw.Draw(img)
        image.text(((__class__.CREATE_SHAPE[0] - w) / 2, (__class__.CREATE_SHAPE[1] - h) / 2), string, font=Text_Font,
                   fill="black")
        img = np.array(img)
        cv2.rectangle(img, (60, 60), (__class__.CREATE_SHAPE[0] - 60, __class__.CREATE_SHAPE[1] - 60), 0, thickness=2)
        return img

    def distorse(img, function=None, axis='x', alpha=1.0, beta=1.0):
        # can use with color or gray scale image
        # example code

        # x directional distorsion
        # img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)

        # y directional distorsion
        # img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)

        # both x and y directional distorsion
        # img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)
        # img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)

        # function are 'sine', 'linear' and 'inverse linear'

        if function != None:
            if function == 'sine':
                A = img.shape[0] / alpha
                w = beta / img.shape[1]

                dist_func = lambda x: A * np.sin(2.0 * np.pi * x * w)
            elif function == 'linear':
                dist_func = lambda x: alpha * x + beta
            elif function == 'inv_linear':
                dist_func = lambda x: -alpha * x - beta
            if axis == 'x':
                for i in range(img.shape[1]):
                    img[:, i] = np.roll(img[:, i], int(dist_func(i)))
            elif axis == 'y':
                for i in range(img.shape[0]):
                    img[i, :] = np.roll(img[i, :], int(dist_func(i)))
        return img


    def crop_image(img,msk,tol=0):
        # img is image data
        # tol  is tolerance
        mask = msk>tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def Get_Plate2(org,thres_kirnel=21,min_area=0.01,max_area=0.9,lengPercent=0.01,morph=False):
        image = copy.deepcopy(org)
        image = __class__.binarize(image,method=__class__.SAUVOLA_THRESHOLDING,value=thres_kirnel)
        image_area = image.shape[0]*image.shape[1]
        if morph:
            image = __class__.morph(image,mode=__class__.ERODE,value=[5,5])

        img, contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        plate = []
        for i in range(0,len(contours)):
            cnt = contours[i]
            hi = hierarchy[0][i]
            epsilon = lengPercent*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            area = cv2.contourArea(cnt)
            if (area>min_area*image_area) and (area < image_area*max_area)and(len(approx) == 4) and(hi[2]!=-1)and(hi[1]==-1):
                plate.append(approx)
                cv2.drawContours(org, [approx], -1, (255, 255, 255), 2)

        for i in range(0,len(plate)):
            plate[i] = np.array(plate[i])
            plate[i] = np.reshape(plate[i],(4,2))
            plate[i] = __class__.four_point_transform(org,plate[i])

        return plate

    def Get_Word2(plate,thres_kirnel=21,boundary=20,black_tollerance=10,image_size=(60,30)):
        listOfWord = []
        for i in range(0,len(plate)):
            word = __class__.binarize(plate[i],method=__class__.SAUVOLA_THRESHOLDING,value=thres_kirnel)

            wx,wy = word.shape
            bou = boundary
            word = 255-np.array(word)
            word = word[bou:wx-bou,bou:wy-bou]

            plate[i] = plate[i][bou:wx-bou,bou:wy-bou]

            #word = IP.morph(word,mode=IP.OPENING,value=[5,5])
            word = __class__.crop_image(plate[i],word,tol=black_tollerance)

            if word != []:
                word = cv2.resize(word,(image_size[1],image_size[0]))
                listOfWord.append(word)
        return listOfWord

    def magnifly(image, percentage=100, shiftxy=[0, 0]):
        # can use with color or gray scale image
        # example code
        # img = IP.magnifly(img,150,shiftxy=[-30,-50])

        # percentage control how big/small the output image is.
        # shiftxy is with respect to top left conner

        try:
            x, y, c = image.shape
        except:
            x, y = image.shape
        x_ = x * percentage // 100
        y_ = y * percentage // 100

        img = cv2.resize(image, (y_, x_),interpolation=cv2.INTER_LANCZOS4)
        base = np.ones((x, y)) * 255.0
        base = Image.fromarray(base)

        img = Image.fromarray(img)
        base.paste(img, (-(x_ - x) // 2 + shiftxy[0], -(y_ - y) // 2 + shiftxy[1]))
        fuck=np.array(base, dtype=np.uint8)
        # base.show('hello')
        return np.array(base, dtype=np.uint8)

    '''IP = Image_Processing_And_Do_something_to_make_Dataset_be_Ready()
        img = cv2.imread('one.jpg',0)
        img = IP.distorse(img,function='sine',axis='x',alpha=20,beta=2)
        img = IP.distorse(img,function='sine',axis='y',alpha=20,beta=2)
        cv2.imshow('img',img)
        cv2.waitKey(0)'''

    def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = __class__.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def remove_perspective(image, region, shape,org_shape=None,auto_sort= True):
        if org_shape == None:
            org_shape = shape


        #print([region[3], region[1], region[2], region[0]])
        #pts1 = np.float32([region[2], region[3], region[1], region[0]])
        pts2 = np.float32([[0, 0], [org_shape[0], 0], [org_shape[0], org_shape[1]], [0, org_shape[1]]])
        if auto_sort:
            best_pts = []
            min_cost = pow(shape[0]*shape[1],2)
            coss = []
            for i in range(0,4):
                rg = np.reshape(region,(-1,2)).tolist()
                pts_1 = np.array(rg[-i:] + rg[:-i])
                pts_2 = np.array(pts2)
                cost = np.sum(np.abs(pts_1-pts_2))
                coss.append(cost)
                if min_cost >= cost:
                    min_cost = cost
                    best_pts = pts_1
            pts_1 = best_pts
            pts2 = np.float32([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]])

            pts1 = np.float32([[pts_1[0]],[pts_1[1]],[pts_1[2]],[pts_1[3]]])
        else:

            pts2 = np.float32([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]])
            pts1 = np.float32([[region[0]],[region[1]],[region[2]],[region[3]]])
        #print(pts1.tolist())
        #pts1 = np.float32([region[1], region[0], region[3], region[2]])
        #print([[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])
        #print('point from auto shuffling',pts1)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(image, matrix, shape,borderValue=255)
        return img

    class Plate():

        #A class for plate
        def __init__(self, image, cnt, word_cnt,extract_shape):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            word_rect = cv2.minAreaRect(word_cnt)
            word_box = cv2.boxPoints(word_rect)
            word_box = np.int0(word_box)

            # print(word_rect)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color, [box], 0, (0, 0, 255), 2)
            cv2.drawContours(color, [word_box], 0, (0, 255, 0), 2)
            # cv2.imshow("color",color)
            matrix = cv2.getRotationMatrix2D((cx, cy), rect[2], 1)
            # self.UnrotateWord = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.remove_perspective(image,
            #                                                                                                   word_box,
            #                                                                                                 (50, 25))
            # print(matrix)
            #cv2.imshow("suck",image)
            #cv2.waitKey(0)
            self.image = image
            #cv2.imshow("suck",self.image)
            #cv2.waitKey(0)
            self.cnt = cnt
            self.PlateBox = box
            self.Original_Word_Size = word_rect[1]
            self.WordBox = word_box
            self.CenterPlate = [cx, cy]
            self.angle = rect[2]
            # print(word_rect)
            self.word_angle = word_rect[2]
            self.show = color

            #self.UnrotateImg = cv2.warpAffine(image, matrix, image.shape, borderMode=cv2.BORDER_CONSTANT,borderValue=255)

            self.UnrotateImg = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.remove_perspective(image,box,(int(rect[1][0]),int(rect[1][1])))

            #self.UnrotateImg = image

            if word_rect[1][0] > word_rect[1][1]:
                y1 = int(word_rect[1][0] / 2) + int(self.UnrotateImg.shape[0]/2)
                y2 = int(self.UnrotateImg.shape[0]/2) - int(word_rect[1][0] / 2)
                x1 = int(word_rect[1][0] / 2) + int(self.UnrotateImg.shape[1]/2)
                x2 = int((self.UnrotateImg.shape[1]/2) - int(word_rect[1][0] / 2))
            else:
                y1 = int(word_rect[1][1] / 2) + int(self.UnrotateImg.shape[0]/2)
                y2 = int(self.UnrotateImg.shape[0]/2) - int(word_rect[1][1] / 2)
                x1 = int(word_rect[1][1] / 2) + int(self.UnrotateImg.shape[1]/2)
                x2 = int(self.UnrotateImg.shape[1]/2) - int(word_rect[1][1] / 2)
            # y1 = int(word_rect[1][0] / 2 + word_rect[0][0])
            # y2 = int(word_rect[0][0] - word_rect[1][0] / 2)
            # x1 = int(word_rect[1][1] / 2 + word_rect[0][1])
            # x2 = int(word_rect[0][1] - word_rect[1][1] / 2)
            # print([x2,x1,y2,y1])
            # print(cx,cy)
            #cv2.imshow("show",self.UnrotateImg)
            #cv2.waitKey(0)
            #print('image',self.UnrotateImg.shape, x2,x1,y2,y1)

            self.UnrotateWord = cv2.resize(self.UnrotateImg[y2:y1, x2:x1], extract_shape)
            # cv2.imshow("kkkkk",self.UnrotateWord)
            # cv2.waitKey(100)
            self.UnrotateWord = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.Adapt_Image(self.UnrotateWord,extract_shape)
            # cv2.imshow("suk",self.UnrotateWord)
            # cv2.waitKey(0)

    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    def get_plate(image,extract_shape,dilate=30):
        org = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image1 = 255 - image
        image1 = __class__.morph(image1, __class__.DILATE, [dilate,dilate])
        #cv2.imshow('image',image1)
        #cv2.waitKey(0)
        img, contours, hierarchy = cv2.findContours(image1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        all_plate = []
        cv2.drawContours(org,contours,-1,[255,0,0])
        # cv2.imshow("jjjj",org)
        #try:
        if 1:
            for cnt, hier, i in zip(contours, hierarchy[0], range(3)):

                if hier[3] == -1:
                    hierach = hier
                    index = 0
                    while hierach[2] != -1:
                        index = hierach[2]
                        hierach = hierarchy[0, index]
                    all_plate.append(__class__.Plate(image, cnt, contours[index],extract_shape))
            return all_plate
        #except:
            #print('return None')
            #return None

    # example
    # img = cv2.imread('ThreeEN.jpg',0)
    # cv2.imshow('org',img)
    # img = Image_Processing_And_Do_something_to_make_Dataset_be_Ready.ztretch(img,axis='horizontal',percentage=0.6)
    # cv2.imshow('result',img)
    # cv2.waitKey(0)


    # morph image acording to mode and value use to construct kernel
    # region value is box containing data that u want to remove perspective like [()]
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''

    def ztretch(image,bord=0,axis='horizontal',multiply=1):
        y,x = image.shape
        bod = [0,0]
        if axis == 'horizontal':

            bod[1] = bord
        elif axis == 'vertical':
            bod[0] = bord
        #image = cv2.resize(image,(multiply*x,multiply*y))
        #point = np.array([(y_//2)-(y//2),(y_//2)+(y//2),(x_//2)-(x//2),(x_//2)+(x//2)])
        #point = point*multiply
        #conner = np.array([[point[0],point[2]],[point[1],point[2]],[point[1],point[3]],[point[0],point[3]]])
        #conner = np.array([[0,0],[x*multiply,0],[x*multiply,y*multiply],[0,y*multiply]])
        #conner = np.array([[0,0],[-y,0],[-y,-x],[0,-x]])
        #print('conner',conner)
        #image = __class__.remove_perspective(image,conner,shape=(y,x),org_shape=(multiply*x,multiply*y))
        #four_point_transform(image,conner,size=(multiply*y,multiply*x),divider=multiply)

        #print(x,y,x_,y_)
        if bod[0] > y//2:
            bod[0] = y//2
        if bod[1] > x//2:
            bod[1] = x//2
        interest = image[bod[0]:y-bod[0]+1,bod[1]:x-bod[1]+1]
        #print('bood after',bod)
        #print('inter',interest.shape)

        return image

    def Adapt_Image(image,output_shape):
        #output_shape =(60,30) #
        ''' (width,height) of picture'''
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        dilate_kernel_shape=(10,10)
        '''2d (x,y) can adjust offset if too less can't extract'''

        inv_image = 255 - image
        dilate = cv2.dilate(inv_image, np.ones(dilate_kernel_shape))
        col = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        # cv2.imshow("dil",dilate)
        # cv2.waitKey(0)
        ret, cnt, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(col,cnt,-1,[0,255,0])
        # cv2.imshow("con",col)
        # print(len(cnt))
        # cv2.waitKey(0)
        try:
            if len(cnt)>0:
                rect = cv2.minAreaRect(cnt[0])
                # print(rect)
                # print(rect)
                if rect[1][0] > rect[1][1]:
                    y1 = int(rect[1][0] / 2 + rect[0][0])
                    y2 = int(rect[0][0] - rect[1][0] / 2)
                    x1 = int(rect[1][1] / 2 + rect[0][1])
                    x2 = int(rect[0][1] - rect[1][1] / 2)
                else:
                    y1 = int(rect[1][1] / 2) + int(rect[0][1])
                    y2 = int(rect[0][1]) - int(rect[1][1] / 2)
                    x1 = int(rect[1][0] / 2) + int(rect[0][0])
                    x2 = int(rect[0][0]) - int(rect[1][0] / 2)
                # print(x2, x1, y2, y1)
                if y2<0 and y1-y2>40:
                    y1 = int(rect[1][1] / 2) + int(rect[0][0])
                    y2 = int(rect[0][0]) - int(rect[1][1] / 2)
                    x1 = int(rect[1][0] / 2) + int(rect[0][1])
                    x2 = int(rect[0][1]) - int(rect[1][0] / 2)
                if(x2>x1):
                    a=x2
                    x2=x1
                    x1=a
                if (y2>y1):
                    a = y2
                    y2 = y1
                    y1 = a
                if x1>output_shape[1]and x2>0 :
                    a= y1
                    b= x2
                    x2 = y2
                    y2 = b
                    y1= x1
                    x1 = a
                if x2<0:
                    x2=0
                if x1>output_shape[1]:
                    x1=output_shape[1]

                # print(x2,x1,y2,y1)
                # y1 = int(rect[1][0] / 2 + rect[0][0])
                # y2 = int(rect[0][0] - rect[1][0] / 2)
                # x1 = int(rect[1][1] / 2 + rect[0][1])
                # x2 = int(rect[0][1] - rect[1][1] / 2)
                img = cv2.resize(image[x2:x1, y2:y1], output_shape)
                # cv2.imshow("img",img)
                # cv2.waitKey(0)
                # ret,img = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
                return img
            else:
                return image
        except:
            return image


    def zkeleton(img,multi=2,morph=15):
        img = 255-img
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False
        size = np.size(img)/multi
        skel = np.zeros(img.shape,np.uint8)
        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True
        skel = 255-skel
        img = __class__.morph(skel,__class__.ERODE,[morph,morph])
        return img


    # extract plate from image
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''




