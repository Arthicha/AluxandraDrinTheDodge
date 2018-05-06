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
CAM_LEFT_FOUR_POINTS = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Right
CAM_RIGHT_PORT = 2
CAM_RIGHT_MODE = 1
CAM_RIGHT_ORIENTATION = -0
CAM_RIGHT_FOUR_POINTS = np.array([[93,81],[116,450],[523,560],[559,81]])

# Camera Bottom Middle
CAM_BOTTOM_MIDDLE_PORT = 3
CAM_BOTTOM_MIDDLE_MODE = 1
CAM_BOTTOM_MIDDLE_ORIENTATION = -90
CAM_BOTTOM_MIDDLE_FOUR_POINTS = np.array([[479, 289], [635, 495], [14, 506], [168, 294]])

# Camera Bottom Right
CAM_BOTTOM_RIGHT_PORT = 4
CAM_BOTTOM_RIGHT_MODE = 1
CAM_BOTTOM_RIGHT_ORIENTATION = -90
CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM = np.array([[339,161],[268,606],[543,155],[554,642]])
CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])

# Camera Bottom Left
CAM_BOTTOM_LEFT_PORT = 5
CAM_BOTTOM_LEFT_MODE = 1
CAM_BOTTOM_LEFT_ORIENTATION = 90
CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM = np.array([[290,230],[356,638],[96,225],[63,639]])
CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT = np.array([[0, 0], [300, 300], [0, 300], [300, 0]])



cam1 = Camera_left(CAM_LEFT_PORT, CAM_LEFT_ORIENTATION, CAM_LEFT_MODE, CAMERA_ALL_OFFSET_Z, CAM_LEFT_FOUR_POINTS)
cam2 = Camera_right(CAM_RIGHT_PORT, CAM_RIGHT_ORIENTATION, CAM_RIGHT_MODE, CAMERA_ALL_OFFSET_Z, CAM_RIGHT_FOUR_POINTS)
cam3 = Camera_Bottom_middle(CAM_BOTTOM_MIDDLE_PORT, CAM_BOTTOM_MIDDLE_ORIENTATION, CAM_BOTTOM_MIDDLE_MODE,
                            CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_MIDDLE_FOUR_POINTS)
cam4 = Camera_Bottom_right(CAM_BOTTOM_RIGHT_PORT, CAM_BOTTOM_RIGHT_ORIENTATION, CAM_BOTTOM_RIGHT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_RIGHT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_RIGHT_FOUR_POINTS_RIGHT)
cam5 = Camera_Bottom_left(CAM_BOTTOM_LEFT_PORT, CAM_BOTTOM_LEFT_ORIENTATION, CAM_BOTTOM_LEFT_MODE,
                           CAMERA_ALL_OFFSET_Z, CAM_BOTTOM_LEFT_FOUR_POINTS_BOTTOM, CAM_BOTTOM_LEFT_FOUR_POINTS_LEFT)