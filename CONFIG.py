from module.Retinutella_theRobotEye import *

'''**************************************************************
**********                                             **********
**********              CAMERA PARAMETER               **********
**********                                             **********
**********                                             **********
**************************************************************'''
# Camera Left
CAM_LEFT_NAME = 'L'
CAM_LEFT_PORT = 1
CAM_LEFT_MODE = 1
CAM_LEFT_ORIENTATION = -90
CAM_LEFT_FOUR_POINTS = np.array([[81, 0], [81, 639], [493, 530], [603, 85]])
CAM_LEFT_MINIMUM_AREA = 0.01
CAM_LEFT_MAXIMUM_AREA = 0.9
CAM_LEFT_LENGTH_PERCENT = 0.01
CAM_LEFT_THRESH_KERNEL = 21
CAM_LEFT_BOUNDARY = 20
CAM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_LEFT_OFFSET_HOMO_X = -81  # -300
CAM_LEFT_OFFSET_HOMO_Y = 0  # -100

# Camera Right
CAM_BOTTOM_MIDDLE_NAME = 'R'
CAM_BOTTOM_MIDDLE_PORT = 2
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = 90
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[559, 4], [560, 639], [145, 504], [42, 77]])
CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.01
CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 21
CAM_BOTTOM_MIDDLE_BOUNDARY = 20
CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -42  # -300
CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = 0  # -100

# Camera Right
CAM_RIGHT_NAME = 'R'
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 90
CAM_RIGHT_FOUR_POINTS = np.array([[559, 4], [560, 639], [145, 504], [42, 77]])
CAM_RIGHT_MINIMUM_AREA = 0.01
CAM_RIGHT_MAXIMUM_AREA = 0.9
CAM_RIGHT_LENGTH_PERCENT = 0.01
CAM_RIGHT_THRESH_KERNEL = 21
CAM_RIGHT_BOUNDARY = 20
CAM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_RIGHT_OFFSET_HOMO_X = -42  # -300
CAM_RIGHT_OFFSET_HOMO_Y = 0  # -100

# Camera Bottom Right
CAM_BOTTOM_RIGHT_NAME = 'Br_bottom'
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS = np.array([[113, 167], [308, 181], [356, 631], [68, 638]])
CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.15
CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
CAM_BOTTOM_RIGHT_BOUNDARY = 20
CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68  # -50
CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167  # -100

# Camera Bottom Left
CAM_BOTTOM_LEFT_NAME = 'Bl_bottom'
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS = np.array([[348, 147], [547, 138], [549, 629], [269, 599]])
CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.87
CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.08
CAM_BOTTOM_LEFT_THRESH_KERNEL = 37
CAM_BOTTOM_LEFT_BOUNDARY = 20
CAM_BOTTOM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -269  # -300
CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -138  # -100

cam_left = Retinutella(CAM_LEFT_NAME, CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAM_LEFT_FOUR_POINTS,
                       CAM_LEFT_THRESH_KERNEL, CAM_LEFT_MINIMUM_AREA, CAM_LEFT_MAXIMUM_AREA, CAM_LEFT_LENGTH_PERCENT,
                       CAM_LEFT_BOUNDARY, CAM_LEFT_BINARIZE_METHOD)

cam_right = Retinutella(CAM_RIGHT_NAME, CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAM_RIGHT_FOUR_POINTS,
                        CAM_RIGHT_THRESH_KERNEL, CAM_RIGHT_MINIMUM_AREA, CAM_RIGHT_MAXIMUM_AREA,
                        CAM_RIGHT_LENGTH_PERCENT, CAM_RIGHT_BOUNDARY, CAM_RIGHT_BINARIZE_METHOD)

cam_bottom_middle = Retinutella(CAM_BOTTOM_MIDDLE_NAME, CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION,
                                CAM_BOTTOM_MIDDLE_MODE, CAM_BOTTOM_MIDDLE_FOUR_POINTS, CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
                                CAM_BOTTOM_MIDDLE_MINIMUM_AREA, CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
                                CAM_BOTTOM_MIDDLE_LENGTH_PERCENT, CAM_BOTTOM_MIDDLE_BOUNDARY,
                                CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)

cam_bottom_left = Retinutella(CAM_BOTTOM_LEFT_NAME, CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION,
                              CAM_BOTTOM_LEFT_MODE, CAM_BOTTOM_LEFT_FOUR_POINTS, CAM_BOTTOM_LEFT_THRESH_KERNEL,
                              CAM_BOTTOM_LEFT_MINIMUM_AREA, CAM_BOTTOM_LEFT_MAXIMUM_AREA,
                              CAM_BOTTOM_LEFT_LENGTH_PERCENT, CAM_BOTTOM_LEFT_BOUNDARY, CAM_BOTTOM_LEFT_BINARIZE_METHOD)

cam_bottom_right = Retinutella(CAM_BOTTOM_RIGHT_NAME, CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION,
                               CAM_BOTTOM_RIGHT_MODE, CAM_BOTTOM_RIGHT_FOUR_POINTS, CAM_BOTTOM_RIGHT_THRESH_KERNEL,
                               CAM_BOTTOM_RIGHT_MINIMUM_AREA, CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
                               CAM_BOTTOM_RIGHT_LENGTH_PERCENT, CAM_BOTTOM_RIGHT_BOUNDARY,
                               CAM_BOTTOM_RIGHT_BINARIZE_METHOD)



# new
# CAMERA_ALL_OFFSET_Z = 25
#
# # Camera Left
# CAM_LEFT_PORT = 1
# CAM_LEFT_MODE = 1
# CAM_LEFT_ORIENTATION = -90
# CAM_LEFT_FOUR_POINTS = np.array([[81, 0], [81, 639], [493, 530], [603, 85]])
# CAM_LEFT_MINIMUM_AREA = 0.01
# CAM_LEFT_MAXIMUM_AREA = 0.9
# CAM_LEFT_LENGTH_PERCENT = 0.01
# CAM_LEFT_THRESH_KERNEL = 21
# CAM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
# CAM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
# CAM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
# CAM_LEFT_THRESH_KERNEL_ORIGINAL = 21
# CAM_LEFT_BOUNDARY = 20
# CAM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
# CAM_LEFT_OFFSET_HOMO_X = -81#-300
# CAM_LEFT_OFFSET_HOMO_Y = 0#-100
#
# # Camera Right
# CAM_RIGHT_PORT = 2
# CAM_RIGHT_MODE = 1
# CAM_RIGHT_ORIENTATION = 90
# CAM_RIGHT_FOUR_POINTS =np.array([[559, 4], [560, 639], [145, 504], [42, 77]])
# CAM_RIGHT_MINIMUM_AREA = 0.01
# CAM_RIGHT_MAXIMUM_AREA = 0.9
# CAM_RIGHT_LENGTH_PERCENT = 0.01
# CAM_RIGHT_THRESH_KERNEL = 21
# CAM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
# CAM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
# CAM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
# CAM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
# CAM_RIGHT_BOUNDARY = 20
# CAM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
# CAM_RIGHT_OFFSET_HOMO_X = -42#-300
# CAM_RIGHT_OFFSET_HOMO_Y = 0#-100
#
# # Camera Bottom Middle
# CAM_BOTTOM_MIDDLE_PORT = 3
# CAM_BOTTOM_MIDDLE_MODE = 1
# CAM_BOTTOM_MIDDLE_ORIENTATION = -180
# CAM_BOTTOM_MIDDLE_FOUR_POINTS =  np.array([[17, 483], [178, 293], [485, 285], [637, 479]])
# CAM_BOTTOM_MIDDLE_MINIMUM_AREA = 0.01
# CAM_BOTTOM_MIDDLE_MAXIMUM_AREA = 0.9
# CAM_BOTTOM_MIDDLE_LENGTH_PERCENT = 0.03
# CAM_BOTTOM_MIDDLE_THRESH_KERNEL = 175
# CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL = 0.01
# CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL = 0.9
# CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL = 0.01
# CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL = 175
# CAM_BOTTOM_MIDDLE_BOUNDARY = 10
# CAM_BOTTOM_MIDDLE_BINARIZE_METHOD = -1
# CAM_BOTTOM_MIDDLE_OFFSET_HOMO_X = -17
# CAM_BOTTOM_MIDDLE_OFFSET_HOMO_Y = -285
#
# # Camera Bottom Right
# CAM_BOTTOM_RIGHT_PORT = 4
# CAM_BOTTOM_RIGHT_MODE = 1
# CAM_BOTTOM_RIGHT_ORIENTATION = -90
# CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[113, 167], [308, 181], [356, 631], [68, 638]])
# # np.array([[119, 173], [51, 639], [358, 638], [314, 189]])
# # np.array([[302,237],[106,230],[65,638],[358,638]])
# CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])
# CAM_BOTTOM_RIGHT_MINIMUM_AREA = 0.01
# CAM_BOTTOM_RIGHT_MAXIMUM_AREA = 0.9
# CAM_BOTTOM_RIGHT_LENGTH_PERCENT = 0.15
# CAM_BOTTOM_RIGHT_THRESH_KERNEL = 37
# CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL = 0.01
# CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL = 0.9
# CAM_BOTTOM_RIGHT_LENGTH_PERCENT_ORIGINAL = 0.01
# CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL = 21
# CAM_BOTTOM_RIGHT_BOUNDARY = 20
# CAM_BOTTOM_RIGHT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
# CAM_BOTTOM_RIGHT_OFFSET_HOMO_X = -68#-50
# CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y = -167#-100
#
# # Camera Bottom Left
# CAM_BOTTOM_LEFT_PORT = 5
# CAM_BOTTOM_LEFT_MODE = 1
# CAM_BOTTOM_LEFT_ORIENTATION = 90
# CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[348, 147], [547, 138], [549, 629], [269, 599]])
# # np.array([[544, 173], [553, 638], [269, 621], [348, 182]])
# # np.array([[342,145],[267,628],[554,639],[550,143]])
# CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])
# CAM_BOTTOM_LEFT_MINIMUM_AREA = 0.01
# CAM_BOTTOM_LEFT_MAXIMUM_AREA = 0.9
# CAM_BOTTOM_LEFT_LENGTH_PERCENT = 0.15
# CAM_BOTTOM_LEFT_THRESH_KERNEL = 37
# CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL = 0.01
# CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL = 0.9
# CAM_BOTTOM_LEFT_LENGTH_PERCENT_ORIGINAL = 0.01
# CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL = 47
# CAM_BOTTOM_LEFT_BOUNDARY = 20
# CAM_BOTTOM_LEFT_BINARIZE_METHOD = IP.SAUVOLA_THRESHOLDING
# CAM_BOTTOM_LEFT_OFFSET_HOMO_X = -269#-300
# CAM_BOTTOM_LEFT_OFFSET_HOMO_Y = -138#-100
#
# cam1 = Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS,
#                    thresh_kernel=CAM_LEFT_THRESH_KERNEL,
#                    thresh_kernel_original=CAM_LEFT_THRESH_KERNEL_ORIGINAL,
#                    minimum_area=CAM_LEFT_MINIMUM_AREA
#                    , minimum_area_original=CAM_LEFT_MINIMUM_AREA_ORIGINAL,
#                    maximum_area=CAM_LEFT_MAXIMUM_AREA,
#                    maximum_area_original=CAM_LEFT_MAXIMUM_AREA_ORIGINAL,
#                    lengthpercent=CAM_LEFT_LENGTH_PERCENT,
#                    lengthpercent_original=CAM_LEFT_LENGTH_PERCENT_ORIGINAL,
#                    word_boundary=CAM_LEFT_BOUNDARY, binarize_method=CAM_LEFT_BINARIZE_METHOD )
# cam2 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS,
#                     thresh_kernel=CAM_RIGHT_THRESH_KERNEL,
#                     thresh_kernel_original=CAM_RIGHT_THRESH_KERNEL_ORIGINAL,
#                     minimum_area=CAM_RIGHT_MINIMUM_AREA
#                     , minimum_area_original=CAM_RIGHT_MINIMUM_AREA_ORIGINAL,
#                     maximum_area=CAM_RIGHT_MAXIMUM_AREA,
#                     maximum_area_original=CAM_RIGHT_MAXIMUM_AREA_ORIGINAL,
#                     lengthpercent=CAM_RIGHT_LENGTH_PERCENT,
#                     lengthpercent_original=CAM_RIGHT_LENGTH_PERCENT_ORIGINAL,
#                     word_boundary=CAM_RIGHT_BOUNDARY, binarize_method=CAM_RIGHT_BINARIZE_METHOD
#                     )
# cam3 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
#                             CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS,
#                             thresh_kernel=CAM_BOTTOM_MIDDLE_THRESH_KERNEL,
#                             thresh_kernel_original=CAM_BOTTOM_MIDDLE_THRESH_KERNEL_ORIGINAL,
#                             minimum_area=CAM_BOTTOM_MIDDLE_MINIMUM_AREA
#                             , minimum_area_original=CAM_BOTTOM_MIDDLE_MINIMUM_AREA_ORIGINAL,
#                             maximum_area=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA,
#                             maximum_area_original=CAM_BOTTOM_MIDDLE_MAXIMUM_AREA_ORIGINAL,
#                             lengthpercent=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT,
#                             lengthpercent_original=CAM_BOTTOM_MIDDLE_LENGTH_PERCENT_ORIGINAL,
#                             word_boundary=CAM_BOTTOM_MIDDLE_BOUNDARY, binarize_method=CAM_BOTTOM_MIDDLE_BINARIZE_METHOD)
# cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
#                                CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM,
#                                CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT,thresh_kernel=CAM_BOTTOM_RIGHT_THRESH_KERNEL,
#                              thresh_kernel_original=CAM_BOTTOM_RIGHT_THRESH_KERNEL_ORIGINAL,minimum_area=CAM_BOTTOM_RIGHT_MINIMUM_AREA
#                              ,minimum_area_original=CAM_BOTTOM_RIGHT_MINIMUM_AREA_ORIGINAL,maximum_area=CAM_BOTTOM_RIGHT_MAXIMUM_AREA,
#                              maximum_area_original=CAM_BOTTOM_RIGHT_MAXIMUM_AREA_ORIGINAL,lengthpercent=CAM_BOTTOM_RIGHT_LENGTH_PERCENT,
#                                Offset_homo_x=CAM_BOTTOM_RIGHT_OFFSET_HOMO_X,Offset_homo_y=CAM_BOTTOM_RIGHT_OFFSET_HOMO_Y)
#
# cam5 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
#                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT,
#                           thresh_kernel=CAM_BOTTOM_LEFT_THRESH_KERNEL,
#                           thresh_kernel_original=CAM_BOTTOM_LEFT_THRESH_KERNEL_ORIGINAL,
#                           minimum_area=CAM_BOTTOM_LEFT_MINIMUM_AREA
#                           , minimum_area_original=CAM_BOTTOM_LEFT_MINIMUM_AREA_ORIGINAL,
#                           maximum_area=CAM_BOTTOM_LEFT_MAXIMUM_AREA,
#                           maximum_area_original=CAM_BOTTOM_LEFT_MAXIMUM_AREA_ORIGINAL,
#                           lengthpercent=CAM_BOTTOM_LEFT_LENGTH_PERCENT,
#                           Offset_homo_x=CAM_BOTTOM_LEFT_OFFSET_HOMO_X, Offset_homo_y=CAM_BOTTOM_LEFT_OFFSET_HOMO_Y)
