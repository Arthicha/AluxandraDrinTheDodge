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
#np.array([[82, 0], [614, 93], [463, 609], [17, 638]])

# Camera Right
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = 90
CAM_RIGHT_FOUR_POINTS =np.array([[560, 1], [71, 64], [164, 527], [560, 639]])
    #np.array([[637, 5], [38, 29], [156, 560], [573, 639]])
    #np.array([[9, 494], [9, 102], [523, 639], [590, 97]])

# Camera Bottom Middle
CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -180
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[23, 503], [636, 508], [485, 304], [180, 309]])
# np.array([[479, 289], [635, 495], [14, 506], [168, 294]])

# Camera Bottom Right
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[106, 167], [301, 182], [349, 619], [68, 629]])
# np.array([[119, 173], [51, 639], [358, 638], [314, 189]])
# np.array([[302,237],[106,230],[65,638],[358,638]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[275, 238], [320, 1], [542, 564], [356, 638]])

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[353, 153], [277, 588], [556, 613], [553, 143]])
# np.array([[544, 173], [553, 638], [269, 621], [348, 182]])
# np.array([[342,145],[267,628],[554,639],[550,143]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[380, 223], [335, 9], [150, 523], [305, 638]])

cam1 = Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS)
cam2 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)
cam3 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS)
cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)
cam5 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                          CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT)
