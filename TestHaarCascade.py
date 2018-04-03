# ~/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

import cv2
from module.detectCascade import multiCascade

def main():

    inputKey = sys.argv[1:3]
    hc = multiCascade()

    if inputKey == [] or str(inputKey[0]) == 'help' :
        sys.exit('test_run.py [param1] [param2]\nparam1:\t 0 or removeAllCascade \n\t 1 or renewCascade\n\t 2 or testCascade\nparam2:\t HAAR / HOG / LBP\n')

    elif str(inputKey[0]) == '0' or str(inputKey[0]) == 'removeAllCascade' :
        hc.deleteCascadeFile()

    elif str(inputKey[0]) == '1' or str(inputKey[0]) == 'renewCascade' :
        '''remove old cascade files and copy new cascade files.'''

        print('remove old cascade files and copy new cascade files.')
        hc.deleteCascadeFile(feature= [str(inputKey[1])])
        hc.copyCascadeFile(feature= str(inputKey[1]))
        print('test')
        predictAll = hc.testCascade(feature= str(inputKey[1]))
        print(predictAll)

    elif str(inputKey[0]) == '2' or str(inputKey[0]) == 'testCascade':
        '''test cascade accuracy file files.'''

        print('test cascade accuracy files.')
        hc.testCascade(feature= str(inputKey[1]))

if __name__ == '__main__':
    main()