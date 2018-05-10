from module.UniqueCameraClass import *

'''**************************************************************
**********                                             **********
**********              CAMERA PARAMETER               **********
**********                                             **********
**********                                             **********
**************************************************************'''

CAMERA_ALL_OFFSET_Z = 25

# Camera Left
CAM_LEFT_PORT = 1
CAM_LEFT_MODE = 1
CAM_LEFT_ORIENTATION = -90
CAM_LEFT_FOUR_POINTS = np.array([[72, 1], [608, 78], [472, 567], [72, 625]])
CAM_LEFT_MINIMUM_AREA = 0.01
CAM_LEFT_MAXIMUM_AREA = 0.9
CAM_LEFT_LENGTH_PERCENT = 0.01
CAM_LEFT_THRESH_KERNEL = 21
CAM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_LEFT_THRESH_KERNEL_ORIGINAL = 21
CAM_LEFT_BOUNDARY = 20
CAM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_LEFT_OFFSET_HOMO_X = -72#-300
CAM_LEFT_OFFSET_HOMO_Y = -78#-100

# Camera Right
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 90
CAM_RIGHT_FOUR_POINTS =np.array([[560, 1], [71, 64], [164, 527], [560, 639]])
CAM_RIGHT_MINIMUM_AREA = 0.01
CAM_RIGHT_MAXIMUM_AREA = 0.9
CAM_RIGHT_LENGTH_PERCENT = 0.01
CAM_RIGHT_THRESH_KERNEL = 21
CAM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
CAM_RIGHT_BOUNDARY = 20
CAM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_RIGHT_OFFSET_HOMO_X = -71#-300
CAM_RIGHT_OFFSET_HOMO_Y = 0#-100

# Camera Bottom Middle
CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -180
CAM_BOTTOM_MIDDLE_FOUR_POINTS =  np.array([[20, 512], [636, 510], [488, 305], [182, 309]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.03
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 175
CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL = 175
CAM_BOTTOM_MIDDLE_BOUNDARY = 10
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = -1
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -20
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = -305

# Camera Bottom Right
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[106, 167], [301, 182], [349, 619], [68, 629]])
# np.array([[119, 173], [51, 639], [358, 638], [314, 189]])
# np.array([[302,237],[106,230],[65,638],[358,638]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])
CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
CAM_BOTTOM_RIGHT_BOUNDARY = 20
CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68#-50
CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167#-100

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[353, 153], [277, 588], [556, 613], [553, 143]])
# np.array([[544, 173], [553, 638], [269, 621], [348, 182]])
# np.array([[342,145],[267,628],[554,639],[550,143]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])
CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_LEFT_THRESH_KERNEL = 37
CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
CAM_BOTTOM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL = 47
CAM_BOTTOM_LEFT_BOUNDARY = 20
CAM_BOTTOM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -277#-300
CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -143#-100

cam1 = Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS,
                   thresh_kernel=CAM_LEFT_THRESH_KERNEL,
                   thresh_kernel_original=CAM_LEFT_THRESH_KERNEL_ORIGINAL,
                   minimum_area=CAM_LEFT_MINIMUM_AREA
                   , minimum_area_original=CAM_LEFT_MINIMUM_AREA_ORIGINAL,
                   maximum_area=CAM_LEFT_MAXIMUM_AREA,
                   maximum_area_original=CAM_LEFT_MAXIMUM_AREA_ORIGINAL,
                   lengthpercent=CAM_LEFT_LENGTH_PERCENT,
                   lengthpercent_original=CAM_LEFT_LENGTH_PERCENT_ORIGINAL,
                   word_boundary=CAM_LEFT_BOUNDARY, binarize_method=CAM_LEFT_BINARIZE_METHOD )
cam2 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS,
                    thresh_kernel=CAM_RIGHT_THRESH_KERNEL,
                    thresh_kernel_original=CAM_RIGHT_THRESH_KERNEL_ORIGINAL,
                    minimum_area=CAM_RIGHT_MINIMUM_AREA
                    , minimum_area_original=CAM_RIGHT_MINIMUM_AREA_ORIGINAL,
                    maximum_area=CAM_RIGHT_MAXIMUM_AREA,
                    maximum_area_original=CAM_RIGHT_MAXIMUM_AREA_ORIGINAL,
                    lengthpercent=CAM_RIGHT_LENGTH_PERCENT,
                    lengthpercent_original=CAM_RIGHT_LENGTH_PERCENT_ORIGINAL,
                    word_boundary=CAM_RIGHT_BOUNDARY, binarize_method=CAM_RIGHT_BINARIZE_METHOD
                    )
cam3 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS,
                            thresh_kernel=CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                            thresh_kernel_original=CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL,
                            minimum_area=CAM_BOTTOM_MIDDLE_MINIMUM_AREA
                            , minimum_area_original=CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL,
                            maximum_area=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                            maximum_area_original=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL,
                            lengthpercent=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT,
                            lengthpercent_original=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL,
                            word_boundary=CAM_BOTTOM_MIDDLE_BOUNDARY, binarize_method=CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)
cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                               CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
                               CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT,thresh_kernel=CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                             thresh_kernel_original=CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_RIGHT_MINIMUM_AREA
                             ,minimum_area_original=CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                             maximum_area_original=CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_RIGHT_LENGTH_PERCENT,
                               Offset_homo_x=CAM_BOTTOM_RIGHT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y)

cam5 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                          CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT,
                          thresh_kernel=CAM_BOTTOM_LEFT_THRESH_KERNEL,
                          thresh_kernel_original=CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL,
                          minimum_area=CAM_BOTTOM_LEFT_MINIMUM_AREA
                          , minimum_area_original=CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL,
                          maximum_area=CAM_BOTTOM_LEFT_MAXIMUM_AREA,
                          maximum_area_original=CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL,
                          lengthpercent=CAM_BOTTOM_LEFT_LENGTH_PERCENT,
                          Offset_homo_x=CAM_BOTTOM_LEFT_OFFSET_HOMO_X, Offset_homo_y=CAM_BOTTOM_LEFT_OFFSET_HOMO_Y)