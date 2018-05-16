import sys
import time
from math import degrees, radians, pi, sqrt, pow
import copy

import cv2
import numpy as np
import serial


from module.MANipulatorKinematics import MANipulator
from module.serialCommu import serial_commu
from module.prePackage import prePackage

class sendSerial:

    def __init__(self,port=4,checkLaser = False, runMatlab= True, sendSerial= True, enLightPos = [[0,500,800],[-250,500,750],[250,500,750]],
                 pathPlaning = 3, initial_position = [200,200,200], recieveSerial= True ,half_IK=False, manualStep = False, 
                 platePositionX= 600, platePositionY = [300,100,-100,-300], platePositionZ = [700,500,300], 
                 offsetLenghtIn = 20, plateHeight = 50, offsetQ = [205,35,150,0,0,0], new_z_equation = 10,
                gainQ = [-1,1,1,1,1,1],modeFixData = False, stepRotation = 5,offsetLenghtOutBottom = 40, offsetLenghtOutOther = 20, servoPlaning = True, 
                offsetBacklash = [0,0,0,0,0,0],caseBacklash = [90,90,90,135,135,135], gainMagnetic = 7/9, qForBackLash= [], 
                planingStepDistance = 10.0, extraoffsetIn = 60,extraoffsetOut = 60, stepOffsetDistance= 10.0, simulator=False):
    
        self.platePositionX = platePositionX
        self.platePositionY = platePositionY
        self.platePositionZ = platePositionZ
        self.offsetLenghtIn = offsetLenghtIn
        self.offsetlenghtOutBottom = offsetLenghtOutBottom
        self.offsetLenghtOutOther = offsetLenghtOutOther
        self.extraoffsetIn = extraoffsetIn
        self.extraoffsetOut = extraoffsetOut
        self.plateHeight = plateHeight

        self.stepRotation = stepRotation

        self.checkLaser = checkLaser

        self.offsetQ = offsetQ
        self.gainQ = gainQ
        self.gainMagnetic = gainMagnetic
        self.offsetBacklash = offsetBacklash
        self.caseBacklash = caseBacklash
        self.qForBackLash = qForBackLash

        self.runMatLab = runMatlab

        self.sendSerial = sendSerial

        self.pathPlaning = pathPlaning
        self.servoPlaning = servoPlaning
        self.planingStepDistance = planingStepDistance
        self.stepOffsetDistance = stepOffsetDistance

        self.recieveSerial = recieveSerial
        self.manualStep = manualStep

        self.half_IK = half_IK
        self.new_z_equation = new_z_equation

        self.initial_position = initial_position
        self.final_position = self.initial_position
        self.modeFixData = modeFixData

        self.enLightPos = enLightPos
        self.oldAns = [radians(90),radians(135),radians(-130),radians(0),radians(0),radians(0)]
        '''-----------------------------------------------------------------------------'''

        self.ser = serial_commu(port=port, sendSerial=self.sendSerial, manualStep= self.manualStep)

        self.MAN = MANipulator()
        self.simulator = simulator
        # self.R_e = MAN.RE_R
        self.package = prePackage(pathPlaning=self.pathPlaning, runMatLab=self.runMatLab, offsetlenghtIn=self.offsetLenghtIn, offsetlenghtOutOther= self.offsetLenghtOutOther,
                                    plateHeight=self.plateHeight ,platePositionX=self.platePositionX, stepOffsetDistance= self.stepOffsetDistance,
                                    platePositionY =self.platePositionY, platePositionZ=self.platePositionZ, extraoffsetIn =self.extraoffsetIn, 
                                    stepRotation= self.stepRotation,offsetlenghtOutBottom=self.offsetlenghtOutBottom, servoPlaning = self.servoPlaning,
                                    enLightPos= enLightPos, planingStepDistance= self.planingStepDistance, extraoffsetOut =self.extraoffsetOut)

        self.ser.clearSerialData()


    def getXYZAndWrite(self,data):
        '''param data : [[position (list), wall (string), pred (int), orentation(3x3 matrix)],...] '''
        listAns = []
        if self.modeFixData == False:
            # data = position wall predict orentation
            data = self.package.make10PathLine(dataList= data) 
            # data = [position wall valve orentation]

            data= self.package.sortBestPosition(dataList= data,initial_position=self.initial_position  ,final_position=self.final_position )
            # data = position wall valve orentation

        # data = position wall valve orentation
        
        # print(data) 
        # input(len(data))
        worseCase = True
        for position,wall,valve,orentation in data:
            print('------------\nget position : '+str([int(po) for po in position]))
            x,y,z = position
            
            R_e = orentation
            dd = self.MAN.d6*R_e[:,2]

            r = sqrt(pow(x,2)+pow(y-85,2)+pow(z-500,2) )
            
            print('r :',r)
            if str(type(self.new_z_equation)) == "<class 'function'>":
                z -= self.new_z_equation(r)                
            
                
            for enLightPos in self.enLightPos:
                # print('sum')
                # input(abs(sqrt(sum([ (enLightPos[0]-x)**2,(enLightPos[1]-y)**2,(enLightPos[2]-z )**2 ] ))))
                
                    
                if abs(sqrt(sum([ (enLightPos[0]-x)**2,(enLightPos[1]-y)**2,(enLightPos[2]-z )**2 ] ))) < 50 and enLightPos != self.enLightPos[1] and worseCase :
                    indexStart = data.index([position,wall,valve,orentation])
                    indexEnd = copy.deepcopy(indexStart)    
                    # input(data.index([position,wall,valve,orentation]))
                    
                    try :
                        while data[indexEnd][2] == valve :
                            # print('ends')
                            # input(indexEnd)
                            indexEnd +=1            
                        indexEnd+=1
                    except:
                        pass
                    # input(indexEnd)
                    worseCase = False
                    break
            try :
                
                dx = x-dd[0]
                dy = y-dd[1]
                dz = z-dd[2]

                ans = self.MAN.inverse_kinamatic2(dx,dy,dz,self.MAN.DH_param,R_e)
                ans = self.MAN.setJointLimits(ans,self.MAN.jointLimit)
                # input(self.enLightPos)

                

                if len(ans) == 0:
                    if self.manualStep:
                        input('joint limit block all')
                if len(ans) >= 1:
                    key = []
                    saveSumVal = 0
                    new_ans = copy.deepcopy(ans)
    
                    for dataList in new_ans:
                        sumVal = 0
                        for indexData in range(len(dataList)):
                            sumVal += abs(dataList[indexData]-self.oldAns[indexData] )
                        if sumVal < saveSumVal or saveSumVal == 0 :
                            saveSumVal = sumVal
                            key = dataList

                    ans = key
                    # for n in range(3,4):
                    #     if abs(ans[n] - self.oldAns[n]) > np.pi/3:
                    #         ans[n] = radians(135)
                    
                    self.oldAns = copy.deepcopy(ans)

                    if self.half_IK:
                        ans = ans[:3]+[0/180*pi,0/180*pi,0/180*pi]
                        
                    
                    nextValve = data[data.index([position,wall,valve,orentation])+1][2]
                    if wall in ['L','R']:
                        
                        if valve == 0 and nextValve == 1:
                            
                            if ans[0] < radians(140):
                                ans[0] -= radians(4)
                            else : 
                                ans[0] += radians(4)
                    
                    listAns.append([ans,valve,position])

                    # print('qqqqqqqqqqqq')
                    # print([position,wall,valve,orentation])
                    # input(data[indexEnd])
                    # print(indexStart)
                    # input(indexEnd)
                    if data[indexEnd] == [position,wall,valve,orentation] and indexEnd != indexStart :
                        for n in range(len(listAns)-(indexEnd-indexStart),len(listAns) ):
                            listAns[n][0][3:6] = copy.deepcopy(listAns[-1][0][3:6])
                            # print(listAns[-1][0][3:6])
                            # print(listAns[n][0][3:6])
                            # input(listAns)
                            # print(n)
                            # print(listAns[-1][0][3:6])
                            # input(listAns[n][0][3:6])

                        worseCase= True
                    
                    # self.decreaseSingularity(self.oldAns,ans,valve)

                
                    # self.getSetQAndWrite(ans,valve)
                    
            except :
                print('IK calculate fail')
                pass
                # sys.exit('IK calculate fail')
        # input('write')
        for ans,valve,position in listAns:
            print('position:',position)
            self.getSetQAndWrite(ans,valve)

                
    def getSetQAndWrite(self,set_q,valve):
        '''param set_Q : [q1,q2,q3,q4,q5,q6]\n\t
             valve : 1 -> open valve, 0 -> closed valve '''
        oldValve = 0
        if self.runMatLab :
            boxHit,laserHit = self.package.boxbreak(set_q)
            if boxHit ==0:
                # ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, offset=MAN.offset, valve=0)
                sys.exit('BOX HIT!!!')
            
            if laserHit == 1 :
                if  self.checkLaser:
                    # ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, offset=MAN.offset, valve=0)
                    sys.exit('ALERT LASER!!!')
                else :
                    print('ALERT LASER!!!')

        if self.simulator:
            H,Hi = self.MAN.forward_kin(self.MAN.DH_param,set_q)
            H_a = [Hi[:,:4,3]]
            x,y,z = [int(i) for i in  Hi[-1,0:3,3]]
            self.MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])

        new_set_q = copy.deepcopy(set_q)
        
        new_set_q = self.reMagnetic(new_set_q,gain = self.gainMagnetic)
        new_set_q = self.offsetBackLash(new_set_q)

        set_q_after_offset = [sum(q) for q in zip( [qi[0]*qi[1] for qi in zip(new_set_q,self.gainQ)] ,[radians(qi) for qi in self.offsetQ] )]

        # print('\nposition per joint : ') 
        # for index_Hi in range(Hi.shape[0]):
        #     xi, yi ,zi = [ int(i) for i in Hi[index_Hi,0:3,3]]
        #     print('joint '+str(index_Hi+1) + ' : ' + str((xi,yi,zi)))

        # print('\nFK position : ' +str((x,y,z)))
        # print('set_q unoffset : '+str([qi for qi in set_q]))
        # print('set_q unoffset : '+str([degrees(qi) for qi in set_q]))
        # print('set_q backlash : '+str([qi/pi for qi in new_set_q]))
        # print('set_q backlash : '+ str( [degrees(qi) for qi in new_set_q]))
        # n = []
        # if oldValve ==  0 and valve == 1:
        #     oldValve =1 
        #     n = set_q_after_offset
        #     if n[0] < radians(140):
        #         n[0] -= radians(4) 
        #     else:
        #         n[0] += radians(4) 
        #     self.ser.write(q=n,valve=valve)        
        # elif oldValve == 1 and valve == 0:
        #     oldValve = 0
        # if n == []:

        self.ser.write(q=set_q_after_offset,valve=valve)        
        
        
        
        if self.recieveSerial:
            serRead = self.ser.read() 
            while serRead != 'A' :
                serRead = self.ser.read() 
                # print(serRead)
                time.sleep(0.1)
                # pass
        
        return 0

    def offsetBackLash(self, set_q):
        
        dataToFunction = [0, set_q[1], 0, 0, 0 ,0]
        caseBacklash = [self.caseBacklash[indexN](dataToFunction[indexN]) for indexN in range(6) ]
        offsetBacklash = [self.offsetBacklash[indexN](self.qForBackLash[0](set_q)) for indexN in range(6)] 
        output = set_q
        for indexQ in range(len(set_q)):
            if set_q[indexQ] > radians(caseBacklash[indexQ]):
                output[indexQ] = set_q[indexQ]-radians(offsetBacklash[indexQ]) 
            else :
                output[indexQ] = set_q[indexQ]+radians(offsetBacklash[indexQ])
        return output

    def reMagnetic(self, set_q,gain = 7/9):
        set_q[0] = ((set_q[0]- radians(90))*gain ) + radians(90)
        return set_q

    def decreaseSingularity(self, oldQ, newQ, valve):
        diffQ = []
        for n in range(3,6):
            diffQ.append(newQ[n]-oldQ[n])
        # input(diffQ)
        if sum(diffQ) > 0.5*pi:
            
            self.getSetQAndWrite([oldQ[0], oldQ[1], oldQ[2], oldQ[3], 0, oldQ[5] ],valve)
            time.sleep(1)
            self.getSetQAndWrite([oldQ[0], oldQ[1], oldQ[2], newQ[3], 0, newQ[5] ],valve)
            time.sleep(1)

        return 0