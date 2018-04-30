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
        '''
        :param method: Method use to binarize the image (to be safe use grayscale image)
                        OTSU_THRESHOLDING 
                        ADAPTIVE_CONTRAST_THRESHOLDING 
                        NIBLACK_THRESHOLDING
                        SAUVOLA_THRESHOLDING
                        BERNSEN_THRESHOLDING
                        LMM_THRESHOLDING
        :param value: Value use to tune Binarization
                        OTSU_THRESHOLDING                   None (can't tune)
                        ADAPTIVE_CONTRAST_THRESHOLDING      [sliding_window_size,weight]    in int and float
                        NIBLACK_THRESHOLDING                [window_size , weight]          in int and float
                        SAUVOLA_THRESHOLDING                window_size                     in int
                        BERNSEN_THRESHOLDING                None (can't tune)
                        LMM_THRESHOLDING                    None (can't tune)
        :return: grayscale image value is 0 or 255  datatype is np.uint8
        :example:
            from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
            img = cv2.imread('one.jpg',0)
            img = ipaddr.binarize(img,method=ipaddr.SAUVOLA_THRESHOLDING , value=35)
            cv2.imshow('img',img)
            cv2.waitKey(0)
        '''
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

    def remove_noise(image, method=KUWAHARA, value=5):
        '''
        :param method:  Method use to filter the image (to be safe use grayscale image)
                        KUWAHARA
                        WIENER
                        MEDIAN 
        :param value: Value use to adjust filter 
                        KUWAHARA            window_size                 in int
                        WIENER              [window_size,how many time] in int
                        MEDIAN              window_size                 in int
        :return: image that have already pass the filter
        :example:
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                img = cv2.imread('one.jpg',0)
                img = ipaddr.remove_noise(img,method=ipaddr.KUWAHARA , value=5)
                cv2.imshow('img',img)
                cv2.waitKey(0)
        '''
        if method == __class__.KUWAHARA:
            img = Filter.Filter_Kuwahara(image, value)
        elif method == __class__.WIENER:
            img = Filter.Filter_weiner(image, window_size=value[0], iteration=value[1])
        elif method == __class__.MEDIAN:
            img = Filter.Filter_median(image, window=value)
        else:
            sys.exit("Unknown method\n")
        return img


    def resize(image, shape=SHAPE, method=INTER_LINEAR):
        '''
        :param image: image to be resize
        :param shape:  shape of output image
                        (x,y) int     
        :param method:  method to approximate value
                    INTER_LINEAR 
                    INTER_AREA 
                    INTER_CUBIC 
                    INTER_NEAREST
                    INTER_LANCZOS4 
                    INTER_MAX
                    WARP_FILL_OUTLIERS 
                    WARP_INVERSE_MAP
        :return: image of size shape
        :example:
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                img = cv2.imread('one.jpg',0)
                img = ipaddr.resize(img,(28,28) )
                cv2.imshow('img',img)
                cv2.waitKey(0)
        '''
        img = cv2.resize(image, shape, interpolation=method)
        return img


    def capture(cam, size=SHAPE, mode=GRAY_SCALE, config=None):
        '''
        :param cam:     cv2 camera object
        :param size:    shape of capture image
                        [x,y] int
        :param mode:    capture as Gray_scale BGR or whatever
                            GRAY_SCALE 
                            RGB 
                            HSV 
                            BGR 
        :param config: additional configuration Well in case you really need it
        :return: capture image as specify color code 
        :example:
            from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
            camera = cv2.videoCapture(0)
            img = ipaddr.capture( camera)
            cv2.imshow('img',img)
            cv2.waitKey(0)
        '''
        ret, image = cam.read()
        if mode != __class__.BGR:
            image = cv2.cvtColor(image, mode, config)
        image = cv2.resize(image, size)
        return image

    def translate(image, value, config=None):
        '''
        :param image:   image to be translate
        :param value:   distance to be translate in pixel
                        (x,y)  int
        :param config:   additional configuration Well in case you really need it
        :return:        image after translate
        :example: 
            from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
            img = cv2.imread('one.jpg',0)
            img = ipaddr.translate(img,(1,1))
            cv2.imshow('img',img)
            cv2.waitKey(0)
        '''
        matrix = np.float32([[1, 0, value[0]], [0, 1, value[1]]])
        if config is None:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]))
        elif config[1] == __class__.BORDER_CONSTANT:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]), flags=config[0],
                                 borderMode=config[1], borderValue=config[2])
        else:
            img = cv2.warpAffine(image, dst=None, M=matrix, dsize=(image.shape[1],image.shape[0]), flags=config[0],
                                 borderMode=config[1])
        return img#

    def blur(image, method=AVERAGING, value=5):
        '''
        :param image :  image to be blur to be sure use grayscale
        :param method:  method of blurring
                         GAUSSIAN 
                         AVERAGING
                         BILATERAL
        :param value:    use to adjust blur
                        GAUSSIAN            n (dimension of kernel  result kernel will be (kernel n*n)/n^2)
                        AVERAGING           n (dimension of kernel  result kernel will be (kernel n*n)/n^2)
                        BILATERAL           [d	Diameter of each pixel neighborhood that is used during filtering. 
                                                If it is non-positive, it is computed from sigmaSpace.  ,
                                            sigmaColor	Filter sigma in the color space. A larger value of the parameter
                                             means that farther colors within the pixel neighborhood (see sigmaSpace)
                                              will be mixed together, resulting in larger areas of semi-equal color.  ,
                                            sigmaSpace	Filter sigma in the coordinate space. A larger value of the 
                                            parameter means that farther pixels will influence each other as long as 
                                            their colors are close enough (see sigmaColor ). When d>0, it specifies 
                                            the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
                                             to sigmaSpace.]
        :return: blur image
        :example:
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                img = cv2.imread('one.jpg',0)
                img = ipaddr.blur(img,ipaddr.AVERAGING,3)
                cv2.imshow('img',img)
                cv2.waitKey(0)
        '''
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

    def morph(image, mode=DILATE, value=[3, 3]):
        '''
        :param image:   image to be morph
        :param mode:    morphological operation
                        ERODE 
                        DILATE 
                        OPENING 
                        CLOSING 
                        GRADIENT 
                        TOP_HAT 
                        BLACK_HAT 
        :param value:   kernel size 
                        [x,y]
        :return:        image passing operation
        :example :
                        from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                        img = cv2.imread('one.jpg',0)
                        img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
                        cv2.imshow('img',img)
                        cv2.waitKey(0)
        '''
        matrix = np.ones((value[0], value[1]), np.float32)
        img = cv2.morphologyEx(image, mode, matrix)
        return img

    def rotation(image, center_of_rotation, angle):
        '''
        :param image: image to be rotated
        :param center_of_rotation:  center of rotation 
                        [x,y] int in pixel
        :param angle:   angle of rotation in degree 
                        int 
        :return: image after getting rotate
        :example:
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                img = cv2.imread('one.jpg',0)
                img = ipaddr.rotation(img,(img.shape[1]/2,img.shape[2]/2),15)
                cv2.imshow('img',img)
                cv2.waitKey(0)
        '''
        matrix = cv2.getRotationMatrix2D((center_of_rotation[0], center_of_rotation[1]), angle, 1)
        img = cv2.warpAffine(image, matrix, (image.shape[1],image.shape[0]), borderMode=__class__.BORDER_CONSTANT,
                             borderValue=255)
        return img


    def font_to_image(font, size=32, index=0, string="0", output_shape = [320,320],border_thickness=2):
        '''
        :param font:    font only trutype 
                        string of font file / path to font
        :param size:    size of character according to font
                        int
        :param index:   index of font some font have many layer
                        int  
        :param output_shape: shape of output image
                            [x,y]
        :param string:  string to create word
                        string/chracter
        :return: image containing border and word
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                image = ipaddr.font_to_image("angsana.ttc", 10, 0, "หนึ่ง")
                cv2.imshow("one", image)
                cv2.waitKey(0)
        '''
        Text_Font = ImageFont.truetype(font, size, index, encoding="unic")
        w, h = Text_Font.getsize(string)
        img = Image.new("L",output_shape , color=255)
        image = ImageDraw.Draw(img)
        image.text(((output_shape [0] - w) / 2, (output_shape [1] - h) / 2), string, font=Text_Font,
                   fill="black")
        img = np.array(img)
        cv2.rectangle(img, (60, 60), (output_shape [0] - 60,output_shape [1] - 60), 0, thickness=border_thickness)
        return img

    def distorse(img, function=None, axis='x', alpha=1.0, beta=1.0):
        '''
        :param img:         image to be distorted
        :param function:    distorted function
                            'sine'
                            'linear'
                            'inv_linear'
        :param axis:        axis of distortion
                            'x' or 'y'
        :param alpha:       coefficient of function
        :param beta:        constant add behind function
        :return: distorted img
        :example: 
                from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr
                img = cv2.imread('one.jpg',0)
                img = ipaddr.distorse(img,function='sine',axis='x',alpha=20,beta=2)
                img = ipaddr.distorse(img,function='sine',axis='y',alpha=20,beta=2)
                cv2.imshow('img',img)
                cv2.waitKey(0)
        '''
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
        '''
        :param img: image to be crop
        :param msk: mask use for crop
        :param tol: tolerance
        :return: crop image
        '''
        # img is image data
        # tol  is tolerance
        mask = msk>tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def center_of_Mass(cnt):
        M=cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx,cy)

    def line_intersection(line1, line2):
        # print('*************')
        # print(line1, line2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            xdiff[1]= -1*xdiff[1]
            div=det(xdiff,ydiff)
            # raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)

    def find_center(pts):
        point = __class__.order_points(pts)
        (tl, tr, br, bl) = point
        point = __class__.line_intersection((tl,br),(bl,tr))
        return point

    def find_orient(pts):
        point = __class__.order_points(pts)
        (tl, tr, br, bl) = point
        return -1*np.arctan2([br[1]-bl[1]],[br[0]-bl[0]])


    def Get_Plate2(org,thres_kirnel=21,min_area=0.01,max_area=0.9,lengPercent=0.01,morph=False, center=False, before =False,orientation=False):
        '''
        :param org:             image to extract plate?
        :param thres_kirnel:    dimension of kernel use to binarize 
                                input as n and it will be n*n size (to be sure use odd number)
        :param min_area:        percentage of minimum area of plate float value range from 1.00 to 0.00
        :param max_area:        percentage of maximum area of plate float value range from 1.00 to 0.00
        :param lengPercent:     percentage of arclength float value range from 1.00 to 0.00
                                (use to calculate epsilon)
        :param morph:           boolean if True image will get erode with preset kernel
                                if False nothing happen to image
        :return: list of plate image
        :example:
                     plate = IP.Get_Plate2(image,min_area=0.01)
                    you will get a list of plate image in the image
        '''
        platePos = []
        plateOrientation=[]
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
            '''My code '''
            if orientation:
                plateOrientation.append(__class__.find_orient(plate[i]))

            if center and before:
                platePos.append(__class__.center_of_Mass(plate[i]))
            '''End of my code'''
            plate[i] = __class__.four_point_transform(org,plate[i])
            '''my code'''
            if center and not before:
                pass
        if orientation and center:
            print(plateOrientation)
            return plate, platePos,plateOrientation
        elif orientation:
            return plate, plateOrientation
        elif center:
            return plate, platePos
        else:
            return plate

    def Get_Word2(plate,thres_kirnel=21,boundary=20,black_tollerance=10,image_size=(60,30)):
        '''
        :param plate:              list of image of plate  
        :param thres_kirnel:        dimension of kernel use to binarize 
                                input as n and it will be n*n size (to be sure use odd number) 
        :param boundary:            boundary of word
                                    int value
                                    from boundary to x-boundary and 20to y -20
        :param black_tollerance:    tolerance of intensity consider to be letter 
        :param image_size:          word image size
        :return: list of word image
        '''
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
        '''
        :param image       image to be magnify
        :param percentage: percent to magnifly 
                            input as percent 100 is normal 200 is two time bigger
        :param shiftxy:     offsetof output image
        :return: magnify image with same shape
        :example:
                        img = IP.magnifly(img,150,shiftxy=[-30,-50])
        '''
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

    def order_points(pts):
        '''
        :parameter pts: 4 point
        :return: sorted list of 4 points in order of
                            top-left
                            top-right
                            bottom-right
                            bottom-left
        '''
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

    def four_point_transform(image, pts,matrice = False):
        '''
        :param image: image in which you get your 4 points
        :param pts: 4 points 
        :param matrice: return the tranformation matrix
        :return: image inside 4 poinr region after removing perspective
        '''
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
        if matrice:
            return warped,M
        return warped

    def remove_perspective(image, region, shape,org_shape=None,auto_sort= True):
        '''
        *********** not use anymore **************
        :param region: 
        :param shape: 
        :param org_shape: 
        :param auto_sort: 
        :return: 
        '''
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
        '''***************** Not being use anymore *********************'''
        #A class for plate
        def __init__(self, image, cnt, word_cnt,extract_shape):
            '''***************** Not being use anymore *********************'''
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
        '''
        :param image:   image to use canny detection
        :param sigma:   
        :return: edge image
        '''
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    def get_plate(image,extract_shape,dilate=30):
        #Decapacipate
        '''
        ***************** Not being use anymore *********************
        :param extract_shape: 
        :param dilate: 
        :return: 
        '''
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
        '''
        
        :param bord:        border
        :param axis:        stretch axis 
        :param multiply:    
        :return: 
        '''
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
        '''
        
        :param output_shape: 
        :return: 
        '''
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
        '''
        :param img:
        :param multi: 
        :param morph: 
        :return: 
        '''
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

    def getHog(listOfImage):
        '''
        :param listOfImage: a list of image          
        :return: a list of histogram of gradient of image
        '''
        def deskew(img):
            m = cv2.moments(img)
            if abs(m['mu02']) < 1e-2:
                # no deskewing needed.
                return img.copy()
            # Calculate skew based on central momemts.
            skew = m['mu11'] / m['mu02']
            # Calculate affine transform to correct skewness.
            M = np.float32([[1, skew, -0.5 ** skew], [0, 1, 0]])
            # Apply affine transform
            img = cv2.warpAffine(img, M, (60, 30), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            return img

        def HOG_int():
            winSize = (20, 20)
            blockSize = (10, 10)
            blockStride = (5, 5)
            cellSize = (10, 10)
            nbins = 9
            derivAperture = 1
            winSigma = -1.
            histogramNormType = 0
            L2HysThreshold = 0.2
            gammaCorrection = 1
            nlevels = 64
            signedGradient = True
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
            return hog
        hog = HOG_int()
        hog_feature = list(map(lambda x: hog.compute(x.astype(np.uint8), winStride=(20, 20)), listOfImage))
        hog_feature = list(map(lambda x: x.reshape((-1,)), hog_feature))
        return  hog_feature

    def getHis(listOfImage,rowOrColPerBar=1):
        '''
        :param listOfImage: a list of image
        :param rowOrColPerBar: a number of row / column that will be group together
                                    ****** must able to divide the row and column of image
        :return: a list of histogram feature
        '''
        histogram_x = list(
            map(lambda x: [x[:, y:y + rowOrColPerBar] for y in range(0,x.shape[1] , rowOrColPerBar)], listOfImage))
        histogram_x = list(map(lambda x: list(map(lambda y: np.sum(y.astype(float)), x)), histogram_x))
        histogram_y = list(
            map(lambda x: [x[y:y + rowOrColPerBar, :] for y in range(0, x.shape[0], rowOrColPerBar)], listOfImage))
        histogram_y = list(map(lambda x: list(map(lambda y: np.sum(y.astype(float)), x)), histogram_y))
        all_histogram = list(map(lambda x, y: np.concatenate([x, y]).tolist(), histogram_x, histogram_y))
        return all_histogram

    def get_XY_feature(point):
        X3 = np.power(point[0], 3)
        XY2 = (point[0]) * (np.power(point[1], 2))
        X5 = np.power(point[0], 5)
        X3Y2 = (np.power(point[0], 3)) * (np.power(point[1], 2))
        XY4 = point[0] * (np.power(point[1], 4))
        X2 = (np.power(point[0], 2))
        Y2 = (np.power(point[1], 2))
        XY = (point[0]) * (point[1])

        # #Y DATAS
        YX2 = (point[1]) * (np.power(point[0], 2))
        Y3 = np.power(point[1], 3)
        YX4 = (point[1]) * (np.power(point[0], 4))
        X2Y3 = (np.power(point[0], 2)) * (np.power(point[1], 3))
        Y5 = np.power(point[1], 5)
        feature_x = np.array([X3, XY2, X5, X3Y2, XY4, X2, Y2, XY])
        feature_y = np.array([YX2, Y3, YX4, X2Y3, Y5, X2, Y2, XY])
        return feature_x,feature_y
    # extract plate from image
    # example
    '''import Image_Processing_And_Do_something_to_make_Dataset_be_Ready() as ipaddr
       img = cv2.imread('one.jpg',0)
       img = ipaddr.morph(img,ipaddr.DILATE,[15,15])
       cv2.imshow('img',img)
       cv2.waitKey(0)'''


