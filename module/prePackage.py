
from math import pow, sqrt
import numpy as np

from module.MATLAPUTOPPU import MATLAPUTOPPU
from module.MotionPlanningZ import point
from module.MANipulatorKinematics import MANipulator


class prePackage:
    def __init__(self,pathPlaning =True,servoPlaning = True,runMatLab=True,extraOfset =60, ofsetlenght=20,ofsetlenght2 = 40, plateHeight=25, platePositionX=[300,100,-100,300], platePositionY =600, platePositionZ=[700,500,300],stepRotation = 5):
        
        self.runMatlab = runMatLab
        if self.runMatlab:
            self.matlab = MATLAPUTOPPU()
        self.pathPlaning = pathPlaning
        self.servoPlaning = servoPlaning

        self.ofsetlenght = ofsetlenght
        self.ofsetlenght2 = ofsetlenght2
        
        Y = platePositionY-int(plateHeight/2)

        self.platePosition = [[platePositionX[0],Y,platePositionZ[0] ],[platePositionX[1],Y,platePositionZ[0] ],
                        [platePositionX[2],Y,platePositionZ[0] ],[platePositionX[3],Y,platePositionZ[0] ],
                        [platePositionX[0],Y,platePositionZ[1] ],[platePositionX[1],Y,platePositionZ[1] ],
                        [platePositionX[2],Y,platePositionZ[1] ],[platePositionX[3],Y,platePositionZ[1] ],
                        [platePositionX[1],Y,platePositionZ[2] ],[platePositionX[2],Y,platePositionZ[2] ] ]
        self.ofsetPlatePosition = [[x,y-ofsetlenght-extraOfset,z] for x,y,z in self.platePosition]
        self.nextOfsetPlatePosition = [[x,y-ofsetlenght2,z] for x,y,z in self.platePosition]
        self.MAN = MANipulator()
        self.stepRotation = stepRotation

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

        keep[(tuple(initial_position[0]),output[0][0])] = [[initial_position[0], keep[output[0]][0][1], 0 , R]  for R in self.getSubRotation(start=initial_position[2],stop = keep[output[0]][0][3] , step = self.stepRotation ) ] + [[data, keep[output[0]][0][1], 0, keep[output[0]][0][3] ] for data in self.sendToPoint(initial_position[0],output[0][0])]
        keep[(output[-1][-1],tuple(final_position[0]))] = [[data, keep[output[-1]][0][1], 0, keep[output[-1]][0][3] ] for data in self.sendToPoint(output[-1][-1],final_position[0])] + [[final_position[0], final_position[1] , 0 , R]  for R in self.getSubRotation(start= keep[output[-1]][0][3],stop = final_position[2] , step = self.stepRotation ) ]
        
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
                priorityOutput.append([realOutput[indexCountRealOutput][0], realOutput[indexCountRealOutput][1], 
                                realOutput[indexCountRealOutput][2], splitSubRotation ])


        # print(priorityOutput)
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

            toDict[pred] = [position,wall,oreintation]
        # sorted toDict and add all required position
        for keyList in sortList: # count pai position
            if keyList in toDict.keys(): #if detect position-number language -> True

                # ofset position
                position,wall,oreintation = toDict[keyList]
                ofsetPosition = [int(val) for val in position]
                nextOfsetPosition = [int(val) for val in position]
                if wall == 'F':
                    ofsetPosition[1] = int(ofsetPosition[1])-self.ofsetlenght
                    nextOfsetPosition[1] = int(nextOfsetPosition[1])-(self.ofsetlenght2)
                if wall == 'L':
                    ofsetPosition[0] = int(ofsetPosition[0])+self.ofsetlenght
                    nextOfsetPosition[0] = int(nextOfsetPosition[0])+(self.ofsetlenght2)
                if wall == 'R':
                    ofsetPosition[0] = int(ofsetPosition[0])-self.ofsetlenght
                    nextOfsetPosition[0] = int(nextOfsetPosition[0])-(self.ofsetlenght2)
                if wall == 'B':
                    ofsetPosition[2] = int(ofsetPosition[2])+self.ofsetlenght   
                    nextOfsetPosition[2] = int(nextOfsetPosition[2])+(self.ofsetlenght2)   

                key = []
                # ofset before get pai to get pai 

                for deltaPosition in self.sendToPoint(ofsetPosition,position):
                    key.append([deltaPosition,wall,0,oreintation] )
                # open valve
                key.append([deltaPosition,wall,1,oreintation] )
   
                # get pai to ofset after get pai 
                for deltaPosition in self.sendToPoint(position,nextOfsetPosition):
                    key.append([deltaPosition,wall,1,oreintation] )

                # ofset from get pai to ofset before put pai
                for deltaPosition in self.sendToPoint(nextOfsetPosition,self.ofsetPlatePosition[tagCount]):
                    key.append([deltaPosition,wall,1,oreintation] )
            
                # ofset before put pai to put pai
                for deltaPosition in self.sendToPoint(self.ofsetPlatePosition[tagCount],self.platePosition[tagCount]):
                    key.append([deltaPosition,'F',1,self.MAN.RE_F] )
                # off valve
                key.append([deltaPosition,'F',0,self.MAN.RE_F] )

                # putpai to ofset after put pai
                for deltaPosition in self.sendToPoint(self.platePosition[tagCount],self.nextOfsetPlatePosition[tagCount]):
                    key.append([deltaPosition,'F',0,self.MAN.RE_F] )
                output.append(key)
                tagCount +=1
        # print(output)
        return output

    def sendToPoint(self,start,end):
        ofsetNewAxis = [500,300,0]
        newStart = [(start[0]+ofsetNewAxis[0])/10, (start[1]+ofsetNewAxis[1])/10, start[2]/10 ]
        newEnd = [(ofsetNewAxis[0]+end[0])/10, (ofsetNewAxis[1]+end[1])/10, end[2]/10 ]    
        if self.pathPlaning:
            data = [[int(x*10-ofsetNewAxis[0]),int(y*10-ofsetNewAxis[1]),int(z*10)] for x,y,z in point(newStart,newEnd)]
            data.append(list(end))
        else :
            # data = [[int(y*10-ofsetNewAxis[1]),int(-x*10+ofsetNewAxis[0]),int(z*10)] for x,y,z in zip(newStart,newEnd) ]
            data = [list(start)]+[list(end)]
        return data
        
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
