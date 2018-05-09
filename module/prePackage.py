
from math import pow, sqrt
import numpy as np
from copy import deepcopy

from module.MATLAPUTOPPU import MATLAPUTOPPU
from module.MotionPlanningZ import point
from module.MANipulatorKinematics import MANipulator


class prePackage:
    def __init__(self,pathPlaning =3,servoPlaning = True,runMatLab=True,extraoffsetIn =60,extraoffsetOut = 60, offsetlenghtIn=20,
    offsetlenghtOut = 40, plateHeight=25, platePositionX=[300,100,-100,300], platePositionY =600, platePositionZ=[700,500,300],
    stepRotation = 5, enLightPos=[[0,500,800],[-250,500,750],[250,500,750]], planingStepDistance = 10.0):
        
        self.runMatlab = runMatLab
        if self.runMatlab:
            self.matlab = MATLAPUTOPPU()
        self.pathPlaning = pathPlaning
        self.servoPlaning = servoPlaning

        self.offsetlenghtIn = offsetlenghtIn
        self.offsetlenghtOut = offsetlenghtOut
        
        self.plateHeight = plateHeight
        Y = platePositionY

        self.platePosition = [[platePositionX[0],Y,platePositionZ[0] ],[platePositionX[1],Y,platePositionZ[0] ],
                        [platePositionX[2],Y,platePositionZ[0] ],[platePositionX[3],Y,platePositionZ[0] ],
                        [platePositionX[0],Y,platePositionZ[1] ],[platePositionX[1],Y,platePositionZ[1] ],
                        [platePositionX[2],Y,platePositionZ[1] ],[platePositionX[3],Y,platePositionZ[1] ],
                        [platePositionX[1],Y,platePositionZ[2] ],[platePositionX[2],Y,platePositionZ[2] ] ]
        self.offsetPlatePosition = [[x,y-extraoffsetIn,z] for x,y,z in self.platePosition]
        self.nextoffsetPlatePosition = [[x,y-extraoffsetOut,z] for x,y,z in self.platePosition]
        self.MAN = MANipulator()
        self.stepRotation = stepRotation
        self.planingStepDistance = planingStepDistance/10.0

        self.LEF_POS = enLightPos[0]
        self.MID_POS = enLightPos[1]
        self.RIG_POS = enLightPos[2]

    def sortBestPosition(self,dataList,initial_position , final_position ):
        output = [] # list start final
        realOutput = []
        priorityOutput = []
        excepted = []
        keep = {}
        # find first and last position of list and keep in dict
        for datas in dataList:
            
            keep[(tuple(datas[0][0]),tuple(datas[-1][0]))] = datas
        
        # sort nearest data
        keys = list(keep.keys())
        while(len(output) < len(keys) ):    
            nearest = 10e+10
            select = []
            for datas in keys:
                if datas not in output :
                    
                    if output==[]:
                        sumdistance = sqrt(sum([pow(initial_position[0][0]-datas[0][0],2), pow(initial_position[0][1] -datas[0][1],2), pow(initial_position[0][2]-datas[0][2],2) ]))
                    else:
                        sumdistance = sqrt(sum([pow(output[-1][1][0]-datas[0][0],2), pow(output[-1][1][1] -datas[0][1],2), pow(output[-1][1][2]-datas[0][2],2) ]))
                    
                    if nearest > sumdistance and sumdistance != 0 :
                        nearest = sumdistance
                        select = datas
                        oldData = datas
                    
            if len(output) < len(keys) and select != []:
                output.append(select)
            else:
                break    


        # add home and final position

        keep[(tuple(initial_position[0]),output[0][0])] = [[initial_position[0], keep[output[0]][0][1], 0 , R]  for R in self.getSubRotation(start=initial_position[2],stop = keep[output[0]][0][3] , step = self.stepRotation ) ] + [[data, keep[output[0]][0][1], 0, keep[output[0]][0][3] ] for data in self.sendToPoint(initial_position[0],output[0][0],'offset')]
        keep[(output[-1][-1],tuple(final_position[0]))] = [[data, keep[output[-1]][0][1], 0, keep[output[-1]][0][3] ] for data in self.sendToPoint(output[-1][-1],final_position[0],'offset')] + [[final_position[0], final_position[1] , 0 , R]  for R in self.getSubRotation(start= keep[output[-1]][0][3],stop = final_position[2] , step = self.stepRotation ) ]
        
        output.insert(0,(tuple(initial_position[0]),output[0][0]) )
        output.insert(len(output),(output[-1][-1],tuple(final_position[0])) )

        
        # add all sub position in path
        count = 0
        for index in range(1,len(output)-2):
            
            count+=1
            start = output[index+count-1][1]
            end = output[index+count][0]
            output.insert(index+count,(start,end))

            keep[(start,end)] = [[data, keep[output[index]][0][1], keep[output[index]][0][2], keep[output[index+count+1]][0][3] ] for data in self.sendToPoint(start,end)] 

         # connect all path and keep in 1 list 
        for index in output: 
            for data in keep[index]:
                if len(realOutput) == 0:
                    realOutput.append(data)
                elif str(data) != str(realOutput[-1]) : 
                    realOutput.append(data)

        # add rotation sub path in all loop
        for indexCountRealOutput in range(len(realOutput)-1):

            for splitSubRotation in self.getSubRotation(start= realOutput[indexCountRealOutput][3], stop=realOutput[indexCountRealOutput+1][3],step= self.stepRotation):
                
                # append data to package if it not in package
                # if str([realOutput[indexCountRealOutput][0], realOutput[indexCountRealOutput][1], realOutput[indexCountRealOutput][2], splitSubRotation ]) not in [str(key) for key in priorityOutput]:
                if len(priorityOutput) == 0:
                    priorityOutput.append([realOutput[indexCountRealOutput][0], realOutput[indexCountRealOutput][1], 
                                    realOutput[indexCountRealOutput][2], splitSubRotation ])
                
                elif str([realOutput[indexCountRealOutput][0], realOutput[indexCountRealOutput][1], realOutput[indexCountRealOutput][2], splitSubRotation ]) != str(priorityOutput[-1]) :
                    priorityOutput.append([realOutput[indexCountRealOutput][0], realOutput[indexCountRealOutput][1], 
                                    realOutput[indexCountRealOutput][2], splitSubRotation ])

        return priorityOutput


    def make10PathLine(self,dataList ):
        '''param datalist = [[3D-position, wall name, predict_output, oreintation  ],...]'''

        sortList = []   # 0 son zero 1 nung one ... 29 yeesibkaw twenty-nine 
        for i in range(10): 
            sortList.append(i)   #number thai eng
            sortList.append(i+10)
            sortList.append(i+20)

        output = [] # [[position, wall, valve, ang],...]
        tagCount = 0
        
        # list -> dict
        toDict = {}
        
        for  position,wall,pred,oreintation in dataList:
            
            if pred not in toDict:
                toDict[pred] = [position,wall,oreintation]
            else:
                count = 1
                while True:
                    if pred+count not in toDict:
                        toDict[pred+count] = [position,wall,oreintation]
                        break
                    else:
                        count +=1

    
        # sorted toDict and add all required position
        for keyList in sortList: # count pai position
            if keyList in toDict.keys(): #if detect position-number language -> True

                # offset position
                position,wall,oreintation = toDict[keyList]
                position =  [int(val) for val in position]
                offsetPosition =  deepcopy(position)
                nextoffsetPosition = deepcopy(position)
                if wall == 'F':
                    position[1] = int(position[1])-int(self.plateHeight)
                    offsetPosition[1] = int(offsetPosition[1])-int(self.offsetlenghtIn)-int(self.plateHeight)
                    nextoffsetPosition[1] = int(nextoffsetPosition[1])-int(self.offsetlenghtOut)-int(self.plateHeight)
                if wall == 'L':
                    position[0] = int(position[0])+int(self.plateHeight)
                    offsetPosition[0] = int(offsetPosition[0])+int(self.offsetlenghtIn)+int(self.plateHeight)
                    nextoffsetPosition[0] = int(nextoffsetPosition[0])+int(self.offsetlenghtOut)+int(self.plateHeight)
                if wall == 'R':

                    position[0] = int(position[0])-int(self.plateHeight)
                    offsetPosition[0] = int(offsetPosition[0])-int(self.offsetlenghtIn)-int(self.plateHeight)
                    nextoffsetPosition[0] = int(nextoffsetPosition[0])-int(self.offsetlenghtOut)-int(self.plateHeight)
                if wall == 'B':
                    position[2] = int(position[2])+int(self.plateHeight)
                    offsetPosition[2] = int(offsetPosition[2])+int(self.offsetlenghtIn)   +int(self.plateHeight)
                    nextoffsetPosition[2] = int(nextoffsetPosition[2])+int(self.offsetlenghtOut)   +int(self.plateHeight)

                key = []
                # offset before get pai to get pai

                for deltaPosition in self.sendToPoint(offsetPosition,position,'offset'):
                    key.append([deltaPosition,wall,0,oreintation] )
                # open valve
                key.append([deltaPosition,wall,1,oreintation] )
   
                # get pai to offset after get pai 
                for deltaPosition in self.sendToPoint(position,nextoffsetPosition,'offset'):
                    key.append([deltaPosition,wall,1,oreintation] )

                # offset from get pai to offset before put pai
                for deltaPosition in self.sendToPoint(nextoffsetPosition,self.offsetPlatePosition[tagCount]):
                    key.append([deltaPosition,wall,1,oreintation] )
            
                # offset before put pai to put pai
                for deltaPosition in self.sendToPoint(self.offsetPlatePosition[tagCount],self.platePosition[tagCount],'offset'):
                    key.append([deltaPosition,'F',1,self.MAN.RE_F] )
                # off valve
                key.append([deltaPosition,'F',0,self.MAN.RE_F] )

                # putpai to offset after put pai
                for deltaPosition in self.sendToPoint(self.platePosition[tagCount],self.nextoffsetPlatePosition[tagCount],'offset'):
                    key.append([deltaPosition,'F',0,self.MAN.RE_F] )
                output.append(key)
                tagCount +=1
  
        return output

    def sendToPoint(self,start,end,case=''):
        output = []
        offsetNewAxis = [500,300,0] 
        if self.pathPlaning == 1: # path planing wan
            
            newStart = [(start[0]+offsetNewAxis[0])/10, (start[1]+offsetNewAxis[1])/10, start[2]/10 ]
            newEnd = [(offsetNewAxis[0]+end[0])/10, (offsetNewAxis[1]+end[1])/10, end[2]/10 ]    
            
            output.append(start)
            for splitPosition in [[int(x*10-offsetNewAxis[0]),int(y*10-offsetNewAxis[1]),int(z*10)] for x,y,z in point(newStart,newEnd,stepDistance= self.planingStepDistance)]:
                output.append(splitPosition)
                # data.append(list(end))
            
            output.append(list(end))

        elif self.pathPlaning == 2: # path planing zumo
            if case != 'offset':
                output = self.enlightMeTheWay(start,end)
            else :
                output = [start]+[end]

        elif self.pathPlaning == 3 : # path planing zumo in wisa

            if case != 'offset':
                enlight = self.enlightMeTheWay(start,end)

                for listEnlight in range(len(enlight)-1):
                    start = enlight[listEnlight]
                    end = enlight[listEnlight+1]
                    newStart = [(start[0]+offsetNewAxis[0])/10, (start[1]+offsetNewAxis[1])/10, start[2]/10 ]
                    newEnd = [(offsetNewAxis[0]+end[0])/10, (offsetNewAxis[1]+end[1])/10, end[2]/10 ]    
                    
                    output.append(start)
                    for splitPosition in [[int(x*10-offsetNewAxis[0]),int(y*10-offsetNewAxis[1]),int(z*10)] for x,y,z in point(newStart,newEnd,stepDistance= self.planingStepDistance)]:
                        output.append(splitPosition)

                    output.append(list(end))
            else :
                newStart = [(start[0]+offsetNewAxis[0])/10, (start[1]+offsetNewAxis[1])/10, start[2]/10 ]
                newEnd = [(offsetNewAxis[0]+end[0])/10, (offsetNewAxis[1]+end[1])/10, end[2]/10 ]    
                
                output.append(start)
                for splitPosition in [[int(x*10-offsetNewAxis[0]),int(y*10-offsetNewAxis[1]),int(z*10)] for x,y,z in point(newStart,newEnd,stepDistance= self.planingStepDistance)]:
                    output.append(splitPosition)
                output.append(list(end))
        else:
            output = [start]+[end]

        return output

    def boxbreak(self,listQ= [0,0,0,0,0,0]):
        a = [[int(listQ[0])],[int(listQ[1])],[int(listQ[2])],[int(listQ[3])],[int(listQ[4])],[int(listQ[5])]]
        box,laser = self.matlab.callMatFunc('collision_check',a,1)[0]
        return (box,laser)

    def getSubRotation(self, start, stop, step):
        output = []
        subRotation = []

        if str(start) != str(stop):

            if self.servoPlaning:
                    for row in range(0,3):
                        for col in range(0,3):
                            section = (stop[row][col]-start[row][col])/(step-1)
                            if abs(float(section)) != 0.0:
                                subRotation.append(np.arange(start[row][col], stop[row][col]+(section/2), section))

                            else:
                                subRotation.append(np.array([start[row][col]] * step))

                    for listIndex in range(step):
                        key = []
                        for row in range(0,3):
                            for col in range(0,3):
                                key.append(subRotation[3*row+col][listIndex])
                        output.append(np.array(key).reshape(3,3))
            else:
                output.append(start)
                output.append(stop)
        else:
            output.append(start)

        return output

    def enlightMeTheWay(self, initial,final):

        onRight = (initial[0] > 0) and (final[0]>0)
        onLeft = (initial[0] <= 0) and (final[0]<=0)
        metaPoint = [self.MID_POS]
        if onLeft or onRight:
            if onLeft:
                metaPoint = [self.LEF_POS]
            elif onRight:
                metaPoint = [self.RIG_POS]
        else:
            if initial[0] > 0:
                metaPoint = [self.RIG_POS,self.MID_POS,self.LEF_POS]
            else:
                metaPoint = [self.LEF_POS,self.MID_POS,self.RIG_POS]
        return [initial]+metaPoint+[final]
