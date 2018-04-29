
# class prePackage:
def to3D(dataDict,plateHeight = 25,spaces = [-400,600,-500,500,0,1000]):
    
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

def addSubPosition(dataList, ofsetlenght=20, plateHeight=25, platePositionX=600, platePositionY =[300,100,-100,300], platePositionZ=[700,500,300],initial_position = [200,0,400]):
    '''arg datalist = [[3D-position, predict_output, degreee of pai, wall name ],...]'''
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