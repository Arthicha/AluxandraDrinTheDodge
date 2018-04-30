
from math import pow

from module.MATLAPUTOPPU import MATLAPUTOPPU
from module.MotionPlanningZ import point
from module.MANipulatorKinematics import MANipulator


class prePackage:
    def __init__(self,checkLaser = False):
        self.matlab = MATLAPUTOPPU()
        self.checkLaser = checkLaser



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
                        sumdistance = sum([abs(initial_position[0]-datas[0][0]), abs(initial_position[1] -datas[0][1]), abs(initial_position[2]-datas[0][2]) ])
                    else:
                        sumdistance = sum([abs(output[-1][1][0]-datas[0][0]), abs(output[-1][1][1] -datas[0][1]), abs(output[-1][1][2]-datas[0][2]) ])
                    
                    if nearest > sumdistance and sumdistance != 0 :
                        nearest = sumdistance
                        select = datas
                        oldData = datas
                    
            if len(output) < len(keys) and select != []:
                output.append(select)
            else:
                break    

        # add all sub position in path
        keep[(tuple(initial_position),output[0][0])] = [[data, 'F', 0, 0 ] for data in self.sendToPoint(initial_position,output[0][0])]
        keep[(output[-1][-1],tuple(final_position))] = [[data, 'F', 0, 0 ] for data in self.sendToPoint(output[-1][-1],final_position)]

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

    def to3D(self,dataDict,plateHeight = 25,spaces = [-400,600,-500,500,0,1000]):
        
        space = {'x':{'min':spaces[0],'max':spaces[1]},'y':{'min':spaces[2],'max':spaces[3]},'z':{'min':spaces[4],'max':spaces[5]}}
        output = []
        
        for i in dataDict:
            for position,img,ang in dataDict[i]:

                # 2D matrix input
                if len(position) == 2:
                    col,row = position
                    if i == 1:
                        x = space['x']['min'] + col                     # -400,500,1000 --  600,500,1000
                        y = space['y']['max'] - plateHeight             #       |               |
                        z = space['y']['max'] - row                     #       |               |
                        output.append([[x,y,z], img, ang, 'L'])         # -400,500,0    --  600,500,0
                    elif i == 2:
                        x = space['x']['max'] - plateHeight             # 600,500,1000  --  600,-500,1000
                        y = space['y']['max'] - col                     #       |               |
                        z = space['z']['max'] - row                     #       |               |
                        output.append([[x,y,z], img, ang, 'F'])         # 600,500,0     --  600,-500,0
                    elif i == 3:
                        x = space['x']['max'] - col                     # 600,-500,1000 --  -400,-500,1000
                        y = space['y']['min'] + plateHeight             #       |               |
                        z = space['z']['max'] - row                     #       |               |
                        output.append([[x,y,z], img, ang, 'R'])         # 600,-500,0    --  -400,-500,0
                    elif i == 4:
                        x = space['x']['max'] - row                     # 600,500,0     --  600,-500,0
                        y = space['y']['max'] - col                     #       |               |
                        z = space['z']['min'] + plateHeight             #       |               |
                        output.append([[x,y,z], img, ang, 'B'])         # -400,500,0    --  -400,-500,0

        return output

    def make10PathLine(self,dataList, ofsetlenght=20, plateHeight=25, platePositionX=600, platePositionY =[300,100,-100,300], platePositionZ=[700,500,300]):
        '''param datalist = [[3D-position, predict_output, degreee of pai, wall name ],...]'''
        X = platePositionX-int(plateHeight/2)

        platePosition = [[X,platePositionY[0],platePositionZ[0] ],[X,platePositionY[1],platePositionZ[0] ],
                        [X,platePositionY[2],platePositionZ[0] ],[X,platePositionY[3],platePositionZ[0] ],
                        [X,platePositionY[0],platePositionZ[1] ],[X,platePositionY[1],platePositionZ[1] ],
                        [X,platePositionY[2],platePositionZ[1] ],[X,platePositionY[3],platePositionZ[1] ],
                        [X,platePositionY[1],platePositionZ[2] ],[X,platePositionY[2],platePositionZ[2] ], ]
        ofsetPlatePosition = [[x-ofsetlenght,y,z] for x,y,z in platePosition]

        sortList = []   # 0 son zero 1 nung one ... 29 yeesibkaw twenty-nine 
        for i in range(10): 
            sortList.append(i)   #number thai eng
            sortList.append(i+10)
            sortList.append(i+20)

        output = [] # [[position, wall, valve, ang],...]
        tagCount = 0

        # list -> dict
        toDict = {}
        for  position,pred,ang,wall in dataList:
            toDict[pred] = [position,ang,wall]
        
        # sorted toDict and add all required position
        for keyList in sortList: # count pai position
            if keyList in toDict.keys(): #if detect position-number language -> True

                position,ang,wall = toDict[keyList]

                ofsetPosition = [int(val) for val in position]
                if wall == 'F':
                    ofsetPosition[0] = int(ofsetPosition[0])-ofsetlenght
                if wall == 'L':
                    ofsetPosition[1] = int(ofsetPosition[1])-ofsetlenght
                if wall == 'R':
                    ofsetPosition[1] = int(ofsetPosition[1])+ofsetlenght
                if wall == 'D':
                    ofsetPosition[2] = int(ofsetPosition[2])+ofsetlenght   

                key = []
                # get pai
                for deltaPosition in self.sendToPoint(ofsetPosition,position):
                    key.append([deltaPosition,wall,0,ang] )
                # open valve
                key.append([deltaPosition,wall,1,ang] )

                # ofset after get pai 
                for deltaPosition in self.sendToPoint(position,ofsetPosition):
                    key.append([deltaPosition,wall,1,ang] )

                # ofset before put pai 
                for deltaPosition in self.sendToPoint(ofsetPosition,ofsetPlatePosition[tagCount]):
                    key.append([deltaPosition,wall,1,ang] )
            
                # put pai 
                for deltaPosition in self.sendToPoint(ofsetPlatePosition[tagCount],platePosition[tagCount]):
                    key.append([deltaPosition,'F',1,0] )
                # off valve
                key.append([deltaPosition,'F',0,0] )

                # ofset after put pai
                for deltaPosition in self.sendToPoint(platePosition[tagCount],ofsetPlatePosition[tagCount]):
                    key.append([deltaPosition,'F',0,0] )
        
                output.append(key)
                tagCount +=1

        return output

    def sendToPoint(self,start,end):
        ofsetNewAxis = [500,300,0]
        newStart = [(ofsetNewAxis[0]-start[1])/10, (ofsetNewAxis[1]+start[0])/10, start[2]/10 ]
        newEnd = [(ofsetNewAxis[0]-end[1])/10, (ofsetNewAxis[1]+end[0])/10, end[2]/10 ]
        data = [[int(y*10-ofsetNewAxis[1]),int(-x*10+ofsetNewAxis[0]),int(z*10)] for x,y,z in point(newStart,newEnd)]
        return data
        
    def boxbreak(self,listQ= [0,0,0,0,0,0]):
        a = [[int(listQ[0])],[int(listQ[1])],[int(listQ[2])],[int(listQ[3])],[int(listQ[4])],[int(listQ[5])]]
        box,laser = self.matlab.callMatFunc('collision_check',a,1)[0]
        return (box,laser)

    def addSubPosition(self,dataList, ofsetlenght=20, plateHeight=25, platePositionX=600, platePositionY =[300,100,-100,300], platePositionZ=[700,500,300],initial_position = [200,0,400]):
        '''param datalist = [[3D-position, predict_output, degreee of pai, wall name ],...]'''
        oldPosition = initial_position
        X = platePositionX-int(plateHeight/2)

        platePosition = [[X,platePositionY[0],platePositionZ[0] ],[X,platePositionY[1],platePositionZ[0] ],
                        [X,platePositionY[2],platePositionZ[0] ],[X,platePositionY[3],platePositionZ[0] ],
                        [X,platePositionY[0],platePositionZ[1] ],[X,platePositionY[1],platePositionZ[1] ],
                        [X,platePositionY[2],platePositionZ[1] ],[X,platePositionY[3],platePositionZ[1] ],
                        [X,platePositionY[1],platePositionZ[2] ],[X,platePositionY[2],platePositionZ[2] ], ]
        ofsetPlatePosition = [[x-ofsetlenght,y,z] for x,y,z in platePosition]

        sortList = []   # 0 son zero 1 nung one ... 29 yeesibkaw twenty-nine 
        for i in range(10): 
            sortList.append(i)   #number thai eng
            sortList.append(i+10)
            sortList.append(i+20)

        output = [] # [[position, wall, valve, ang],...]
        tagCount = 0

        # list -> dict
        toDict = {}
        for  position,pred,ang,wall in dataList:
            toDict[pred] = [position,ang,wall]
        
        # sorted toDict and add all required position
        for keyList in sortList: # count pai position
            if keyList in toDict.keys(): #if detect position-number language -> True

                position,ang,wall = toDict[keyList]

                ofsetPosition = [int(val) for val in position]
                if wall == 'F':
                    ofsetPosition[0] = int(ofsetPosition[0])-ofsetlenght
                if wall == 'L':
                    ofsetPosition[1] = int(ofsetPosition[1])-ofsetlenght
                if wall == 'R':
                    ofsetPosition[1] = int(ofsetPosition[1])+ofsetlenght
                if wall == 'D':
                    ofsetPosition[2] = int(ofsetPosition[2])+ofsetlenght

                if False : # Use wisaFunction
                    pass
                    # # ofset before get pai
                    # for deltaPosition in wisaFunction(ofsetPosition, oldPosition):
                    #     output.append([deltaPosition,wall,0,ang] )
                    
                    # # get pai
                    # for deltaPosition in wisaFunction(position, ofsetPosition):
                    #     output.append([deltaPosition,wall,1,ang] )

                    # # ofset after get pai 
                    # for deltaPosition in wisaFunction(ofsetPosition, position):
                    #     output.append([deltaPosition,wall,1,ang] )
                
                    # # ofset before put pai position
                    # for deltaPosition in wisaFunction(ofsetPlatePosition[tagCount] , ofsetPosition):
                    #     output.append([deltaPosition,'F',1,0]) 

                    # # put pai position
                    # for deltaPosition in wisaFunction(platePosition[tagCount] , ofsetPlatePosition[tagCount]):
                    #     output.append([deltaPosition,'F',1,0])
            
                    # # ofset after put pai position
                    # for deltaPosition in wisaFunction(ofsetPlatePosition[tagCount] , platePosition[tagCount]):
                    #     output.append([deltaPosition,'F',0,0])
                    
                    # oldPosition = ofsetPlatePosition[tagCount]
                    # tagCount +=1


                else : #  Not use WisaFunction 

                    # ofset before get pai
                    output.append([ofsetPosition,wall,0,ang] )

                    # get pai
                    output.append([position,wall,1,ang] )

                    # ofset after get pai 
                    output.append([ofsetPosition,wall,1,ang] )
                
                    # ofset before put pai position
                    output.append([ofsetPlatePosition[tagCount],'F',1,0])

                    # put pai position
                    output.append([platePosition[tagCount],'F',1,0])
            
                    # ofset after put pai position
                    output.append([ofsetPlatePosition[tagCount],'F',0,0])

                    tagCount +=1

                

        return output