
from math import pow, sqrt

from module.MATLAPUTOPPU import MATLAPUTOPPU
from module.MotionPlanningZ import point
from module.MANipulatorKinematics import MANipulator


class prePackage:
    def __init__(self,pathPlaning =True,runMatLab=True, ofsetlenght=20, plateHeight=25, platePositionX=[300,100,-100,300], platePositionY =600, platePositionZ=[700,500,300]):
        
        self.runMatlab = runMatLab
        if self.runMatlab:
            self.matlab = MATLAPUTOPPU()
        self.pathPlaning = pathPlaning

        self.ofsetlenght = ofsetlenght
        
        Y = platePositionY-int(plateHeight/2)

        self.platePosition = [[platePositionX[0],Y,platePositionZ[0] ],[platePositionX[1],Y,platePositionZ[0] ],
                        [platePositionX[2],Y,platePositionZ[0] ],[platePositionX[3],Y,platePositionZ[0] ],
                        [platePositionX[0],Y,platePositionZ[1] ],[platePositionX[1],Y,platePositionZ[1] ],
                        [platePositionX[2],Y,platePositionZ[1] ],[platePositionX[3],Y,platePositionZ[1] ],
                        [platePositionX[1],Y,platePositionZ[2] ],[platePositionX[2],Y,platePositionZ[2] ] ]
        self.ofsetPlatePosition = [[x,y-ofsetlenght,z] for x,y,z in self.platePosition]
        self.MAN = MANipulator()


    def sortBestPosition(self,dataList,initial_position = [200,0,400], final_position = [200,0,400]):
        output = [] # list start final
        realOutput = []
        excepted = []
        keep = {}
        # find first and last position of list and keep in dict
        for datas in dataList:
            
            keep[(tuple(datas[0][0]),tuple(datas[-1][0]))] = datas
        
        # sort data
        keys = list(keep.keys())
        while(len(output) < len(keys) ):    
            nearest = 10e+10
            select = []
            for datas in keys:
                if datas not in output :
                    
                    if output==[]:
                        sumdistance = sqrt(sum([pow(initial_position[0]-datas[0][0],2), pow(initial_position[1] -datas[0][1],2), pow(initial_position[2]-datas[0][2],2) ]))
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

        # add all sub position in path
        keep[(tuple(initial_position),output[0][0])] = [[data, 'F', 0, self.MAN.RE_F ] for data in self.sendToPoint(initial_position,output[0][0])]
        keep[(output[-1][-1],tuple(final_position))] = [[data, 'F', 0, self.MAN.RE_F ] for data in self.sendToPoint(output[-1][-1],final_position)]

        output.insert(0,(tuple(initial_position),output[0][0]) )
        output.insert(len(output),(output[-1][-1],tuple(final_position)) )
        
        count = 0
        for index in range(1,len(output)-2):
            
            count+=1
            start = output[index+count-1][1]
            end = output[index+count][0]
            output.insert(index+count,(start,end))

            keep[(start,end)] = [[data, keep[output[index]][0][1], keep[output[index]][0][2], keep[output[index]][0][3] ] for data in self.sendToPoint(start,end)] 
         # connect all path and keep in 1 list 
        for index in output: 
            for data in keep[index]:
                realOutput.append(data)

        return realOutput


    def make10PathLine(self,dataList ):
        '''param datalist = [[3D-position, wall name, predict_output, orentation ],...]'''

        sortList = []   # 0 son zero 1 nung one ... 29 yeesibkaw twenty-nine 
        for i in range(10): 
            sortList.append(i)   #number thai eng
            sortList.append(i+10)
            sortList.append(i+20)

        output = [] # [[position, wall, valve, ang],...]
        tagCount = 0
        
        # list -> dict
        toDict = {}
        for  position,wall,pred,orentation in dataList:

            toDict[pred] = [position,wall,orentation]
        # sorted toDict and add all required position
        for keyList in sortList: # count pai position
            if keyList in toDict.keys(): #if detect position-number language -> True

                position,wall,orentation = toDict[keyList]
                ofsetPosition = [int(val) for val in position]
                if wall == 'F':
                    ofsetPosition[1] = int(ofsetPosition[1])-self.ofsetlenght
                if wall == 'L':
                    ofsetPosition[0] = int(ofsetPosition[0])+self.ofsetlenght
                if wall == 'R':
                    ofsetPosition[0] = int(ofsetPosition[0])-self.ofsetlenght
                if wall == 'B':
                    ofsetPosition[2] = int(ofsetPosition[2])+self.ofsetlenght   

                key = []
                # get pai

                for deltaPosition in self.sendToPoint(ofsetPosition,position):
                    key.append([deltaPosition,wall,0,orentation] )
                # open valve
                key.append([deltaPosition,wall,1,orentation] )

                # ofset after get pai 
                for deltaPosition in self.sendToPoint(position,ofsetPosition):
                    key.append([deltaPosition,wall,1,orentation] )

                # ofset before put pai 
                for deltaPosition in self.sendToPoint(ofsetPosition,self.ofsetPlatePosition[tagCount]):
                    key.append([deltaPosition,wall,1,self.MAN.RE_F] )
            
                # put pai 
                for deltaPosition in self.sendToPoint(self.ofsetPlatePosition[tagCount],self.platePosition[tagCount]):
                    key.append([deltaPosition,'F',1,self.MAN.RE_F] )
                # off valve
                key.append([deltaPosition,'F',0,self.MAN.RE_F] )

                # ofset after put pai
                for deltaPosition in self.sendToPoint(self.platePosition[tagCount],self.ofsetPlatePosition[tagCount]):
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
