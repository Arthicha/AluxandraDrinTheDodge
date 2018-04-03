# ~/virtualenv/ROBOTICS_studios/bin/python

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''
import sys
import os

from module.detectCascade import multiCascade

def main():

    inputKey = sys.argv[1:3]
    hc = multiCascade()

    if inputKey == [] or str(inputKey[0]) == 'help' :
        sys.exit('test_run.py [param1] [param2]\nparam1:\t 1 or testCascade\nparam2:\t HAAR / HOG / LBP\n')

    elif str(inputKey[0]) == '1' or str(inputKey[0]) == 'testCascade':
        '''test cascade accuracy file files.'''

        print('test cascade accuracy files.')
        output = hc.testCascade(feature= str(inputKey[1]))
        print(hc.calculateAccuracy(output))
        hc.displayConclusionMatrix(output)
        

if __name__ == '__main__':
    main()
