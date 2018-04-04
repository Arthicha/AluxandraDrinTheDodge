# ~/virtualenv/ROBOTICS_studios/bin/python
# this class for generate cascade classifier files

'''*************************************************
*                                                  *
*             import class & library               *
*                                                  *
*************************************************'''

import os, shutil
import sys
import platform
from random import choice, randrange, shuffle
from time import sleep

from PIL import Image, ImageOps
from math import sqrt, pow
import numpy as np


class trainCascade():
    def __init__(self):
        '''*************************************************
        *                                                     *
        *                 define valuable                  *
        *                                                  *
        *************************************************'''

        if platform.system() == 'Linux':
            self.dirCom = '/'
        elif platform.system() == 'Windows':
            self.dirCom = '\\'
        else :
            self.dirCom = '/'

        self.scaleWeightHeight = 0.5
        self.scalePosNeg = 10
        self.memoryUse = 4096
        self.multiPos = 2

        self.minHitRate = 0.950
        self.maxFalseAlarmRate = 0.200
        self.weightTrimRate = 0.35
        self.maxDepth = 1
        self.maxWeakCount = 80
        self.mode = 'BASIC' # BASIC CORE ALL

        self.CompressDataPath = 'dataset'
        self.trainModelPath = 'savedModel'+self.dirCom+'modelHAAR'
        self.mainTrainSubPath = 'trainModel'
        self.mainExtractDataSetSubPath = 'ExtractData'
        self.mainImageSubPath = 'mainData'
        self.mainOutputSubPath = 'outputData'

        self.mainPosFilePath = (self.trainModelPath+ self.dirCom+ self.mainTrainSubPath+ 
                                self.dirCom+ 'numPos.txt')
        self.mainNegFilePath = (self.trainModelPath+ self.dirCom+ self.mainTrainSubPath+ 
                                self.dirCom+ 'numNeg.txt')
        self.mainExtractDataSetPath = (self.trainModelPath+ self.dirCom+ self.mainTrainSubPath+ 
                                self.dirCom+ self.mainExtractDataSetSubPath)
        self.mainImagePath = (self.trainModelPath+ self.dirCom+ self.mainTrainSubPath+ 
                                self.dirCom+ self.mainImageSubPath)
        self.mainOutputPath = (self.trainModelPath+ self.dirCom+ self.mainTrainSubPath+ 
                                self.dirCom+ self.mainOutputSubPath)


        self.listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six',
                        'seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH',
                        'SevenTH','EightTH','NineTH']
        self.suffix = ['test','train','validate']

        '''*************************************************
        *                                                  *
        *             define anoymous function             *
        *                                                  *
        *************************************************'''

        self.WHfromArray1D = lambda arraySize : ( int(sqrt(arraySize*self.scaleWeightHeight)), 
                                                int(sqrt(arraySize/self.scaleWeightHeight)) )




    '''*************************************************
    *                                                  *
    *                    sub module                    *
    *                                                  *
    *************************************************'''


    def removeOldData(self):
        if os.path.isdir(os.path.join(self.trainModelPath,self.mainTrainSubPath)):
            shutil.rmtree(os.path.join(self.trainModelPath,self.mainTrainSubPath))
        
        return 0

    def createRootDirectoryAndFiles(self):

        if not os.path.isdir(os.path.join(self.trainModelPath,self.mainTrainSubPath)):
            os.mkdir(os.path.join(self.trainModelPath,self.mainTrainSubPath))

        if not os.path.isdir(os.path.join(self.trainModelPath+self.dirCom+
                            self.mainTrainSubPath,self.mainExtractDataSetSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+self.dirCom+
                    self.mainTrainSubPath,self.mainExtractDataSetSubPath))

        if not os.path.isdir(os.path.join(self.trainModelPath+self.dirCom+
                            self.mainTrainSubPath,self.mainImageSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+self.dirCom+
                    self.mainTrainSubPath,self.mainImageSubPath))     

        if not os.path.isdir(os.path.join(self.trainModelPath+self.dirCom+
                            self.mainTrainSubPath,self.mainOutputSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+self.dirCom+
                    self.mainTrainSubPath,self.mainOutputSubPath))  

            for subFolder in self.listOfClass:
                os.mkdir(os.path.join(self.mainOutputPath,str(subFolder)))     


        return 0

    def copyUsedModel(self):
        for filePath in os.listdir(self.mainOutputPath):
            shutil.copy(self.mainOutputPath+self.dirCom+filePath+self.dirCom+
                        'cascade.xml',self.trainModelPath+ self.dirCom+ 
                        str(filePath)+'.xml')

        return 0 

    def settingHyperParameter(self, scaleWeightHeight=0.5, scalePosNeg=10,
                                memoryUse=1024, multiPos=1, minHitRate=0.950,
                                maxFalseAlarmRate=0.5, weightTrimRate=0.95,
                                maxDepth=1, maxWeakCount=100, mode='BASIC'):

        self.scaleWeightHeight = scaleWeightHeight
        self.scalePosNeg = scalePosNeg
        self.memoryUse = memoryUse
        self.multiPos = multiPos

        self.minHitRate = minHitRate
        self.maxFalseAlarmRate = maxFalseAlarmRate
        self.weightTrimRate = weightTrimRate
        self.maxDepth = maxDepth
        self.maxWeakCount =maxWeakCount
        self.mode = mode 
        return 0
        
    def extractPictureAndResize(self, limitFilePerClass = 50, size = 16, 
                                mainPos = 'validate-0.png'):
        '''generate picture from file in folder dataset and save to folder
        dataExtract with limit picture per class.'''

        '''*************************************************
        *                                                  *
        *              config generate number              *
        *                                                  *
        *************************************************'''
        numCount = 0
        numKeep = 0

        '''*************************************************
        *                                                  *
        *              read & generate data                *
        *                                                  *
        *************************************************'''

        for s in range(1,3): # train & validate
            for j in range(0,30): # 30 class
                object = self.listOfClass[j]
                f = open(self.CompressDataPath+self.dirCom+str(j)+'_'+
                        self.suffix[s]+'.txt','r')
                image = str(f.read()).split('\n')[:-1]
                f.close()

                numKeep += numCount
                numCount = 0
                for i in range(len(image)):
                    
                    fileName = (self.dirCom+str(object)+'_'+self.suffix[s]+
                                '-'+str(numCount)+'.png')

                    image[i] = np.fromstring(image[i], dtype=int, sep=',')
                    image[i] = np.array(image[i])
                    
                    image[i] = np.reshape(image[i],self.WHfromArray1D(len(image[i])))
                    img = Image.fromarray((image[i]).astype(np.uint8))
                    img = ImageOps.invert(img) 

                    if img.height < int(size) or img.width < int(size):
                        sys.exit('size is bigger than '+str(img.height)+','+str(img.width))
                    img = img.resize((int(int(size)/self.scaleWeightHeight),int(size)),
                                    Image.ANTIALIAS)

                    img.save(self.mainExtractDataSetPath+fileName)

                    if self.suffix[s]+'-'+str(numCount)+'.png' == mainPos:
                        img.save(self.mainImagePath+fileName)

                    if numCount > int(limitFilePerClass)-1 :
                        break
                    if (numCount%int(int(limitFilePerClass)/2)) == 0 :
                        print("generate"+str(numKeep+numCount)+ ":"+self.suffix[s]+
                                '-'+str(object) +"-"+str(numCount))
                    numCount+=1
        return 0


    def create_bg_txt(self, select_value):
        '''use image from dataExtract and input string to write bg_neg.txt and bg_pos.txt .'''
        
        '''*************************************************
        *                                                  *
        *            remove & create old file              *
        *                                                  *
        *************************************************'''

        f_pos = open(self.mainPosFilePath,"w+")
        f_neg = open(self.mainNegFilePath,"w+")
        
        '''*************************************************
        *                                                  *
        *                 random data list                 *
        *                                                  *
        *************************************************'''
        
        listData = os.listdir(self.mainExtractDataSetPath)
        randomList = []
        while len(listData) > 0 :
            randomData = choice(listData)
            randomList.append(randomData)
            listData.remove(randomData)    
    
        '''*************************************************
        *                                                  *
        *            split positive and negative           *
        *                                                  *
        *************************************************'''

        countPos =0
        countNeg =0
        if str(select_value) in str(self.listOfClass):

            for f in randomList:
                if str(f.split('_')[0]) == str(select_value):

                    f_pos.write('ExtractData'+self.dirCom+f+"\n")
                    countPos+=1

            countNegs = (int(countPos/len(self.listOfClass))*len(self.listOfClass) * 
                            self.scalePosNeg)
            
            keepList = []
            while (countNeg < countNegs):
                
                key = (str(randrange(0,len( [i for i in [ str(j).split('0_train') 
                                                        for j in randomList ]
                                                        if len(i) == 2] )))+'.png')
                
                for selectClass in self.listOfClass:

                    keepList.append('ExtractData'+self.dirCom+str(selectClass)+
                                    '_train-'+str(key)+"\n")
                    countNeg+=1 

            shuffle(keepList)
            for selectList in keepList:
                f_neg.write(selectList)
                    
        else:
            sys.exit('out of class')
        
        f_pos.close()
        f_neg.close()

        print("number of positive : "+str(countPos))
        print("number of negative : "+str(countNeg))



    def run_opencv_createsamples(self, main_class='',number=''):
        ''' opencv_createsamples library from libopencv-dev .\n
            To generate vector file for run opencv_traincascade .'''

        if main_class=='' or number=='':
            sys.exit('main class or number is invalid')

        weight, height = Image.open(self.mainImagePath+self.dirCom+
                                    os.listdir(self.mainImagePath)[0]).size

        img = self.mainImagePath+self.dirCom+str(main_class)+'*'
        vec = os.path.join(self.trainModelPath+self.dirCom+self.mainTrainSubPath,'positives.vec')

        command = ('opencv_createsamples -img '+str(img)+' -bg '+self.mainPosFilePath+' -vec '+
                    vec+ ' -bgcolor 0 -maxxangle 1.2 -maxyangle 1.2 -maxzangle 0.5 -num '+
                    str(number) +' -w '+ str(weight)+' -h '+str(height))
        os.system(command)

    def run_opencv_traincascade(self, main_class='0',numpos=0,numneg=0,numstate=0,
                                feature='HAAR'):
        ''' opencv_traincascade library from libopencv-dev .\n
            To generate haarCascade classification file. '''

        if numpos==0 or numneg==0 or numstate==0 :
            sys.exit('numpos | numneg | numstate is 0')
        
        weight, height = Image.open(self.mainImagePath+self.dirCom+
                                    os.listdir(self.mainImagePath)[0]).size
        

        data = self.mainOutputPath+self.dirCom+str(main_class) +self.dirCom
        vec = self.trainModelPath+self.dirCom+self.mainTrainSubPath+self.dirCom +'positives.vec'
        command = ('opencv_traincascade -featureType '+str(feature)+ ' -mode '+
                    str(self.mode)+ ' -data '+ str(data) +' -vec '+ str(vec) +
                    ' -bg '+self.mainNegFilePath+' -numPos '+
                    str(numpos)+' -numNeg '+str(numneg)+' -numStages '+str(numstate)+
                    ' -w '+str(weight)+' -h '+str(height)+' -minHitRate '+
                    str(self.minHitRate)+' -maxFalseAlarmRate '+
                    str(self.maxFalseAlarmRate)+' -weightTrimRate '+
                    str(self.weightTrimRate)+' -maxDepth '+str(self.maxDepth) +
                    ' -maxWeakCount '+str(self.maxWeakCount) +' -precalcValBufSize '
                    +str(self.memoryUse)+' -precalcIdxBufSize '+str(self.memoryUse))
        
        os.system(command)

    def run_opencv_haartraining(self):
        '''Now, don't know how it use.'''
        
        pass

    def run_opencv_performance(self):
        '''Now, don't know how it use.'''
            
        pass

    def AutoGenerateClassification(self, numberPerClass=1000, main_img='validate-0',
                                    size=16, numstate=10, feature='HAAR'):
        '''auto generate 30 classification by auto parameter.'''

        print('gen_image '+str(numberPerClass)+' per class')
        
        self.removeOldData()
        self.createRootDirectoryAndFiles()
        self.extractPictureAndResize(limitFilePerClass= numberPerClass, size= size,
                                    mainPos= main_img+'.png')
        
        for selectClass in self.listOfClass:
            self.create_bg_txt(select_value=selectClass)
            
            with open(self.mainNegFilePath,'r') as f :
                countNeg = len(str(f.read()).split('\n')[:-1])
            with open(self.mainPosFilePath,'r') as f :
                countPos = len(str(f.read()).split('\n')[:-1])
                
            num = self.predictNumPosNumNeg(countPos=countPos*4/5,countNeg=countNeg*4/5)    
                
            self.run_opencv_createsamples(main_class=selectClass,number=int(countPos))

            self.run_opencv_traincascade(main_class=selectClass,numpos=int(num[0]),
                                        numneg=int(num[1]),numstate=int(numstate),
                                        feature=feature)

            sleep(10)
        self.copyUsedModel()
        self.removeOldData()


    def predictNumPosNumNeg(self, countPos,countNeg):
        ''' find NumPos and NumNeg in term i*pow(10,n) .'''
        countKeep = 0
        pos = int(countPos)
        neg = int(countNeg)
        while pos >= 10:
            pos /= 10
            countKeep+=1
        pos = int(pow(10,countKeep)*int(pos))
        
        countKeep = 0
        while neg >= 10:
            neg /= 10
            countKeep+=1
        neg = int(pow(10,countKeep)*int(neg))
        
        
        # pos = int(pos/100)*100
        # neg = int(neg/100)*100
        

        return [pos,neg]
  


