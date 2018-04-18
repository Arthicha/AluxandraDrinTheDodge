''' :class HaarLikeFeature(model_path) :\n
    :class trainCascade(model_path) :'''

'''*************************************************
*                                                  *
*                  import library                  *
*                                                  *
*************************************************'''

import os
import shutil
import sys
from time import clock, sleep
from random import randrange, shuffle

import numpy as np
import cv2
from PIL import Image, ImageOps
from math import sqrt, pow
import matplotlib.pyplot as plot
from module import FAI

dirSep = os.path.sep

dataSetPath = 'dataset'+dirSep+'synthesis'+dirSep+'textfile_norm'

class HaarLikeFeature:
    '''for detect text from camera with 30 haar-cascade classifier and manage classifier files
        :function settingHyperParameter(): setting hyperparameter\n
        :function predict(): test 30 classifier\n
        :function displayConfusionMatrix(): display matrix to windows'''
        
    def __init__(self, model_path='savedModel'+dirSep+'modelHAAR'):
        '''initial all variable and path'''

        self.modelPath = model_path
        self.datasetPath = dataSetPath

        self.scaleWeightHeight = 0.5
        self.testResizeH = 32

        self.displayWindow = 0
        self.multiClassifiers = []
        self.listOfClass = [0,1,2,3,4,5,6,7,8,9]+['zero','one','two','three','four','five','six','seven','eight','nine']+['ZeroTH','OneTH','TwoTH','ThreeTH','FourTH','FiveTH','SixTH','SevenTH','EightTH','NineTH']
        self.suffix = ['test','train','validate']
        self.scaleFactor = 3.2
        self.minNeighbors = 5
        self.output = np.zeros((30,30),np.uint16)
        '''*************************************************
        *                                                  *
        *             define anoymus function              *
        *                                                  *
        *************************************************'''

        self.WHfromArray1D = lambda arraySize : ( int(sqrt(arraySize*self.scaleWeightHeight)), int(sqrt(arraySize/self.scaleWeightHeight)) )

    def settingHyperParameter(self, scaleFactor=1.2, minNeighbors=4):
        '''
        :param scaleFactor: ...\n
        :param minNeighbors: number of minimium node that specific to image is ?\n
        :return:none\n
        example \n
                multiCascade.settingHyperParameter(scaleFactor=1.2, minNeighbors=4)\n
                this function setting 2 HyperParameter values of testing'''

        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        return 0

    def callClassifiers(self,feature):
        ''' call all classifier \n
            :param feature: define classifier name to test\n
            :return: None \n
            example\n
                    multiCascade.callClassifier(feature='HAAR')\n
                    this function prepare classifier for test'''
        if feature == 'HAAR':
            self.multiClassifiers = {i:cv2.CascadeClassifier(self.modelPath+dirSep+str(i)) for i in os.listdir(self.modelPath) if len(str(i).split('.')) == 2 }
        elif  feature == 'LBP':
            self.multiClassifiers = {i:cv2.CascadeClassifier(self.modelPath+dirSep+str(i)) for i in os.listdir(self.modelPath) if len(str(i).split('.')) == 2}
        return 0

    def detectFromCascade(self,image):
        '''for detect text from camera with 30 cascade classifier\n
            :param image: input for detected with 30 classifier\n
            :return: classifier list that you can detect image\n
            example \n
                    multiCascade.detectFromCascade('test0.png')\n
                    this function will detect this image and return list detect classifier '''

        if self.multiClassifiers == []:
            self.callClassifiers(feature='HAAR')
        img = image
        returnData = {}
        
        for selectClassifier in list(self.multiClassifiers):

            output = self.multiClassifiers[selectClassifier].detectMultiScale(img, scaleFactor= self.scaleFactor,minNeighbors= self.minNeighbors)
            

            output2 = []
            returnData[str(selectClassifier).split('.')[0]] = 0
        
            if len(output) != 0:
                
                for (x, y, w, h) in output :
                    if h*1.2 > img.shape[0] and w*1.2 > img.shape[1]:
                        returnData[str(selectClassifier).split('.')[0]] +=1
                        output2.append((x,y,w,h))
                        if self.displayWindow :
                            print((selectClassifier, w,h))
                            cv2.rectangle(image, (x,y), (x+w,y+h), 0, 1)
                            cv2.imshow('test',image)
                            cv2.waitKey(50)
                            image = img
                        
        return returnData

	    
    def predict(self, data = [], Return = ''):
        ''' test classifier by select feature (Default HAAR).\n
            :param feature: define classifier name to test\n
            :return: Confusion matrix in np 30x30 matrix\n 
            example 
                    multiCascade.testCascade(feature='HAAR')\n
                    this function test cascade by use haar classifier method '''	
        # for suffixSelect in [0] :# ['test','train','validate']

        keepData=np.zeros((len(data),30 ))

        imageCount = 0
        
        self.callClassifiers(feature='HAAR')
        keep =0
            			

        for i in range(len(data)):
            data[i] = np.fromstring(data[i], dtype=int, sep=',')
            data[i] = np.array(data[i], dtype=np.uint8)
            data[i] = np.reshape(data[i],(self.WHfromArray1D(len(data[i]))))
            # data[i] = 255-data[i]
            if data[i].shape[:2] != (int(self.testResizeH/self.scaleWeightHeight),int(self.testResizeH)):
                data[i] = cv2.resize(data[i],(int(self.testResizeH/self.scaleWeightHeight),int(self.testResizeH)))


            detect = self.detectFromCascade(image=data[i])
            
            for det in detect:

                keepData[keep,int(det)] += 1
            keep +=1

        if Return == 'prob':
            return  keepData
        elif Return == 'class':
            return FAI.prob2Class(keepData)
            
    def testCascade(self,feature='HAAR'):
        ''' test classifier by select feature (Default HAAR).\n
        :param feature: define classifier name to test\n
        :return: Confusion matrix in np 30x30 matrix\n 
        example 
                multiCascade.testCascade(feature='HAAR')\n
                this function test cascade by use haar classifier method '''	
        for suffixSelect in [0] :# ['test','train','validate']
            keepData={}
            keepDataAll = {}
            for i in range(0,30): # 30 class
                keepDataAll[i]={}
                for j in range(0,30): # inloop 30 class
                    keepDataAll[i].update({j:0})

            imageCount = 0
            tic = clock()
            
            self.callClassifiers(feature=feature)
            
            for j in range(0,30): # 30 class
                object = self.listOfClass[j]
                f = open( self.datasetPath +dirSep+str(j)+'_'+self.suffix[suffixSelect]+'.txt','r')
                image = str(f.read()).split('\n')[:-1]
                f.close()
                keepData[object] = 0			
                imageCount += len(image)
                tic_n = clock()

                for i in range(len(image)):
                    image[i] = np.fromstring(image[i], dtype=int, sep=',')
                    image[i] = np.array(image[i], dtype=np.uint8)
                    image[i] = np.reshape(image[i],(self.WHfromArray1D(len(image[i]))))
                    # image[i] = 255-image[i]
                    if image[i].shape[:2] != (int(self.testResizeH/self.scaleWeightHeight),int(self.testResizeH)):
                        image[i] = cv2.resize(image[i],(int(self.testResizeH/self.scaleWeightHeight),int(self.testResizeH)))

                    
                    detect = self.detectFromCascade(image=image[i])
                    # keepData[object]+=detect[str(object)]
                    
                    # print (detect)
                    for obj in detect:
                        keepDataAll[j][int(obj)] += int(detect[obj])

                toc_n = clock()

            toc = clock()
        self.output = self.convertDict2matrix( keepDataAll)
        print(imageCount)
        print('accuracy:'+str( sum([keepDataAll[i][i] for i in range(30)])/imageCount )+'/1')
        return self.output    

    def calculateAccuracy(self, dataMatrix= np.zeros((30,30),np.uint8)):
        '''make matrix to accuracy from 30x30 numpy matrix\n
            :param dataMatrix: Confusion matrix that you want to find f1 score\n
            :return: f1 score\n 
            example \n
                    multiCascade.calculateAccuracy(dataMatrix= np.zeros((30,30),np.uint8)) \n
                    this function calculate f1 score and return value \n'''
        FN = 0
        TP = 0
        FP = 0
        TN = 0
        for row_n in range(30):
            for col_n in range(30):
                if row_n == col_n :
                    TP += dataMatrix[row_n][col_n]
                else :
                    FP += dataMatrix[row_n][col_n]
        return ((FN,TP,FP,TN))
    
    def displayConfusionMatrix(self, dataMatrix = np.zeros((30,30),np.uint8)):
        '''generate confusion matrix and display \n
            :param dataMatrix: confusion matrix that youw{0:{0:0,1:0,...,29:0} want to plot show\n
            :return: no return but display \n
            example \n
                    multicascade.displayConfusionMatrix(dataMatrix= np.zeros((30,30),np.uint8))\n
                    this function display matrix 30x30 in 1 window'''
        fig, ax = plot.subplots()

        for row_n in range(len(self.listOfClass )):
            for col_n in range(len(self.listOfClass)):
                
                ax.text(row_n, col_n, str(dataMatrix[col_n][row_n]), va = 'center', ha = 'center')
                
      
        ax.matshow(dataMatrix, cmap = plot.cm.Blues)
        ax.set_xlabel('real')
        ax.set_ylabel('predictions')
        ax.set_xticks(np.arange(30))    
        ax.set_yticks(np.arange(30))
        ax.set_xticklabels([str(i) for i in self.listOfClass], rotation =90)
        ax.set_yticklabels([str(i) for i in self.listOfClass])
        # plot.savefig('output_data'+dirSep+'keep_predict_test'+dirSep+str(name)+'.png')
        plot.show()

    def convertDict2matrix(self, informationDict={}):
        '''convert dict with 30 multidict into np matrix\n
        :param informationDict: dict that keep information to create Confusion matrix\n
        :return: data in 30x30 matrix pattern in type of np.uint16\n
        example \n
                multiCascade.convertDict2matrix({0:{0:0,1:0,...,29:0},...,{0:{0:0,1:0,...,29:0}})\n
                this function will convert dixt into matrix format'''
        returnMat = np.zeros((30,30), np.uint16)
    
        for row_n in range(len(self.listOfClass )):
            for col_n in range(len(self.listOfClass)):
                
                returnMat[row_n][col_n] = informationDict[col_n][row_n]

        return returnMat
        




class trainCascade:
    '''this class is for train cascade classifier\n
        :function settingHyperParameter(): setting hyperparameter\n 
        :function autoGenerateClassification(): auto train 30 cascade classification'''

    def __init__(self, model_path='savedModel'+dirSep+'modelHAAR'):
        '''*************************************************
        *                                                  *
        *                 define valuable                  *
        *                                                  *
        *************************************************'''

        self.runCase = 0 # 0 is normal & other -- 1 is normal & inv -- 2 all 

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
        self.bgcolor = 255

        self.CompressDataPath = dataSetPath
        self.trainModelPath = model_path
        self.mainTrainSubPath = 'trainModel'
        self.mainExtractDataSetSubPath = 'extractData'
        self.mainInvDataSetSubPath = 'invData'
        self.mainImageSubPath = 'mainData'
        self.mainOutputSubPath = 'outputData'
        

        self.mainPosFilePath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ 'numPos.txt')
        self.mainNegFilePath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ 'numNeg.txt')
        self.mainExtractDataSetPath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ self.mainExtractDataSetSubPath)
        self.mainInvDataSetPath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ self.mainInvDataSetSubPath)
        self.mainImagePath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ self.mainImageSubPath)
        self.mainOutputPath = (self.trainModelPath+ dirSep+ self.mainTrainSubPath+ 
                                dirSep+ self.mainOutputSubPath)


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
        '''this function remove all finish used dir \n
            :no param:\n
            :no return:\n
            example\n
                    trainCascade.removeOldData()\n'''
        if os.path.isdir(os.path.join(self.trainModelPath,self.mainTrainSubPath)):
            shutil.rmtree(os.path.join(self.trainModelPath,self.mainTrainSubPath))
        
        return 0

    def createRootDirectoryAndFiles(self):
        '''this function create all require dir \n
            :no param:\n
            :no return:\n
            example\n
                    trainCascade.createRootDirectoryAndFiles()\n'''
        if not os.path.isdir(os.path.join(self.trainModelPath,self.mainTrainSubPath)):
            os.mkdir(os.path.join(self.trainModelPath,self.mainTrainSubPath))

        if not os.path.isdir(os.path.join(self.trainModelPath+dirSep+
                            self.mainTrainSubPath,self.mainExtractDataSetSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+dirSep+
                    self.mainTrainSubPath,self.mainExtractDataSetSubPath))

        if not os.path.isdir(os.path.join(self.trainModelPath+dirSep+
                            self.mainTrainSubPath,self.mainInvDataSetSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+dirSep+
                    self.mainTrainSubPath,self.mainInvDataSetSubPath))

        if not os.path.isdir(os.path.join(self.trainModelPath+dirSep+
                            self.mainTrainSubPath,self.mainImageSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+dirSep+
                    self.mainTrainSubPath,self.mainImageSubPath))     

        if not os.path.isdir(os.path.join(self.trainModelPath+dirSep+
                            self.mainTrainSubPath,self.mainOutputSubPath)):
            os.mkdir(os.path.join(self.trainModelPath+dirSep+
                    self.mainTrainSubPath,self.mainOutputSubPath))  

            for subFolder in self.listOfClass:
                os.mkdir(os.path.join(self.mainOutputPath,str(subFolder)))     


        return 0

    def copyUsedModel(self):
        '''this function copy important model from train dir before delete it \n
            :no param:\n
            :no return:\n
            example\n
                    trainCascade.copyUsedModel()\n'''
        keep =0
        a = {}
        for i in range(30):
            a[str(self.listOfClass[i])] = i
        for filePath in os.listdir(self.mainOutputPath):
            shutil.copy(self.mainOutputPath+dirSep+filePath+dirSep+
                        'cascade.xml',self.trainModelPath+ dirSep+ 
                        str(a[filePath])+'.xml')
        
        return 0

    def settingHyperParameter(self, scaleWeightHeight=0.5, scalePosNeg=10,
                                memoryUse=1024, multiPos=1, minHitRate=0.950,
                                maxFalseAlarmRate=0.5, weightTrimRate=0.95,
                                maxDepth=1, maxWeakCount=100, mode='BASIC'):
        '''this function setting all requirement parameter \n
            :param scaleWeightHeight: scale between width and height \n
            :param scalePosNeg: scale between number of positives and negatives\n
            :param memoryUse: memory use limit\n
            :param multipos: extract feature for create more positive image\n
            :param minHitRate: value of minimum TP train detecting per round\n
            :param maxFalseAlarmRate: value of maximum FP train detecting per round\n
            :param weightTrimRate: difference of train weight per step\n
            :param maxDepth: value for increase dampping train out\n
            :param maxWeakCount: value for limit train step per round\n
            :param mode: mode of training\n  
            :no return:\n
            example\n
                    trainCascade.settingHyperParameter(scaleWeightHeight=0.5, scalePosNeg=10,
                                memoryUse=1024, multiPos=1, minHitRate=0.950,
                                maxFalseAlarmRate=0.5, weightTrimRate=0.95,
                                maxDepth=1, maxWeakCount=100, mode='BASIC')\n'''
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
        dataExtract with limit picture per class.\n
            :param limitFilePerClass: limited image per dataset (set 0 nolimit)\n
            :param size: value of height that you want to resize image\n
            :param mainPos: main image that you select for train classifier\n
            :no return:
            example\n
                    trainCascade.extractPictureAndResize(limitFilePerClass=1200,size=16,mainPos='validate-0.png')'''

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
                f = open(self.CompressDataPath+dirSep+str(j)+'_'+
                        self.suffix[s]+'.txt','r')
                image = str(f.read()).split('\n')[:-1]
                f.close()
                if int(limitFilePerClass)==0:
                    limitFilePerClass = len(image)

                numKeep += numCount
                numCount = 0
                for i in range(len(image)):
                    
                    fileName = (dirSep+str(object)+'_'+self.suffix[s]+
                                '-'+str(numCount)+'.png')

                    image[i] = np.fromstring(image[i], dtype=int, sep=',')
                    image[i] = np.array(image[i])
                    image[i] = np.reshape(image[i],self.WHfromArray1D(len(image[i])))
                    img = Image.fromarray((image[i]).astype(np.uint8))

                    if img.height < int(size) or img.width < int(size):
                        sys.exit('size is bigger than '+str(img.height)+','+str(img.width))
                    img = img.resize((int(int(size)/self.scaleWeightHeight),int(size)),
                                    Image.ANTIALIAS)

                    invimg = ImageOps.invert(img) 
                    img.save(self.mainExtractDataSetPath+fileName)
                    invimg.save(self.mainInvDataSetPath+fileName)

                    if self.suffix[s]+'-'+str(numCount)+'.png' == mainPos:
                        img.save(self.mainImagePath+fileName)

                    if (numCount > int(limitFilePerClass)-1 ) :
                        break
                    if (numCount%int(int(limitFilePerClass)/2)) == 0 :
                        print("generate"+str(numKeep+numCount)+ ":"+self.suffix[s]+
                                '-'+str(object) +"-"+str(numCount))
                    numCount+=1

                if int(limitFilePerClass)==len(image):
                    limitFilePerClass = 0
        return 0


    def create_bg_txt(self, select_value):
        '''use image from dataExtract and input string to write bg_neg.txt and bg_pos.txt .\n
            :param select_value: positive classifier\n
            :return: None\n
            example\n
                    trainCascade.create_bg_txt('1')\n
                    create bg_pos.txt is classificate 1  and bg_neg.txt is other'''
        
        '''*************************************************
        *                                                  *
        *            remove & create old file              *
        *                                                  *
        *************************************************'''

        f_pos = open(self.mainPosFilePath,"w+")
        f_neg = open(self.mainNegFilePath,"w+")
        
        listData = os.listdir(self.mainExtractDataSetPath)
 
        '''*************************************************
        *                                                  *
        *            split positive and negative           *
        *                                                  *
        *************************************************'''

        countPos =0
        countNeg =0

        shuffle(listData)
        if str(select_value) in str(self.listOfClass):
            
            for f in listData:
                if str(f.split('_')[0]) == str(select_value):

                    f_pos.write(self.mainExtractDataSetSubPath+dirSep+f+"\n")
                    countPos+=1

            countNegs = (int(countPos/len(self.listOfClass))*len(self.listOfClass) * 
                            self.scalePosNeg)
            
            keepList = []

            if self.runCase == 0 or self.runCase == 2:
                while (countNeg < countNegs):
                    
                    key = (str(randrange(0,len( [i for i in [ str(j).split('0_train') 
                                                            for j in listData ]
                                                            if len(i) == 2] )))+'.png')
                    
                
                    for selectClass in self.listOfClass:
                        if str(selectClass) != str(select_value): 
                            keepList.append(self.mainExtractDataSetSubPath+dirSep+
                                            str(selectClass)+'_train-'+str(key)+"\n")
                            countNeg+=1

                countNegs+=countNeg
            if self.runCase == 1 or self.runCase == 2:
                shuffle(listData)
                while (countNeg < countNegs):
                    for f in listData :
                        if str(f.split('_')[0]) == str(select_value):
                            keepList.append(self.mainInvDataSetSubPath+dirSep+f+"\n")
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
            To generate vector file for run opencv_traincascade .\n
            this mothod for buildin  '''

        if main_class=='' or number=='':
            sys.exit('main class or number is invalid')

        weight, height = Image.open(self.mainImagePath+dirSep+
                                    os.listdir(self.mainImagePath)[0]).size

        img = self.mainImagePath+dirSep+str(main_class)+'*'
        vec = os.path.join(self.trainModelPath+dirSep+self.mainTrainSubPath,'positives.vec')

        command = ('opencv_createsamples -img '+str(img)+' -bg '+self.mainPosFilePath+' -vec '+
                    vec+ ' -maxxangle 1.2 -maxyangle 1.2 -maxzangle 0.5 -num '+
                    str(number) +' -w '+ str(weight)+' -h '+str(height)+' -bgcolor '+str(self.bgcolor))
        os.system(command)

    def run_opencv_traincascade(self, main_class='0',numpos=0,numneg=0,numstate=0,
                                feature='HAAR'):
        ''' opencv_traincascade library from libopencv-dev .\n
            To generate haarCascade classification file. \n
            this mothod for buildin '''

        if numpos==0 or numneg==0 or numstate==0 :
            sys.exit('numpos | numneg | numstate is 0')
        
        weight, height = Image.open(self.mainImagePath+dirSep+
                                    os.listdir(self.mainImagePath)[0]).size
        

        data = self.mainOutputPath+dirSep+str(main_class) +dirSep
        vec = self.trainModelPath+dirSep+self.mainTrainSubPath+dirSep +'positives.vec'
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


    def AutoGenerateClassification(self, numberPerClass=1000, main_img='validate-0',
                                    size=16, numstate=10, feature='HAAR'):
        '''auto generate 30 classification by auto parameter.\n
            :param numberPerClass: maximum number per dataset (set 0 nolimit)\n
            :param main_img: selected main image for prepare models\n
            :param size: value of height for resize train image\n
            :param numstate: number of classfier state\n
            :param feature: feture that you train\n
            :return: None\n
            example\n
                    trainCascade.AutogenerateClassification(numberPerClass=1000,main_img='validate-0'
                    ,size=16,numstate=5,feature='HAAR') '''

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
                
            # num = self.predictNumPosNumNeg(countPos=countPos*4/5,countNeg=countNeg*4/5)    
                
            self.run_opencv_createsamples(main_class=selectClass,number=int(countPos))

            self.run_opencv_traincascade(main_class=selectClass,numpos=int(countPos*4/5),
                                        numneg=int(countNeg),
                                        numstate=int(numstate), feature=feature)


        self.copyUsedModel()
        self.removeOldData()


    def predictNumPosNumNeg(self, countPos,countNeg):
        ''' find NumPos and NumNeg in term i*pow(10,n)\n
            this method for built in function .'''
        
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
  


