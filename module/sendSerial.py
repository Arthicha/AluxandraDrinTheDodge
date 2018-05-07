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
                 platePositionX= 600, platePositionY = [300,100,-100,-300], platePositionZ = [700,500,300], extraoffset = 60,
                 offsetLenght = 20, plateHeight = 50, workspace = [-400,600,-500,500,0,1000], offsetQ = [205,35,150,0,0,0],
                gainQ = [-1,1,1,1,1,1],modeFixData = False, stepRotation = 5,offsetLenght2 = 40, servoPlaning = True, 
                offsetBacklash = [0,0,0,0,0,0],caseBacklash = [90,90,90,135,135,135] ):
    
        self.platePositionX = platePositionX
        self.platePositionY = platePositionY
        self.platePositionZ = platePositionZ
        self.offsetLenght = offsetLenght
        self.offsetLenght2 = offsetLenght2
        self.extraoffset = extraoffset
        self.plateHeight = plateHeight

        self.stepRotation = stepRotation

        self.checkLaser = checkLaser

        self.workspace = workspace

        self.offsetQ = offsetQ
        self.gainQ = gainQ
        self.offsetBacklash = offsetBacklash
        self.caseBacklash = caseBacklash

        self.runMatLab = runMatlab

        self.sendSerial = sendSerial

        self.pathPlaning = pathPlaning
        self.servoPlaning = servoPlaning

        self.recieveSerial = recieveSerial
        self.manualStep = manualStep

        self.half_IK = half_IK

        self.initial_position = initial_position
        self.final_position = self.initial_position
        self.modeFixData = modeFixData

        '''-----------------------------------------------------------------------------'''

        self.ser = serial_commu(port=port, sendSerial=self.sendSerial, manualStep= self.manualStep)
        input('press reset board and press any key and enter:')
        self.MAN = MANipulator()
        # self.R_e = MAN.RE_R
        self.package = prePackage(pathPlaning=self.pathPlaning, runMatLab=self.runMatLab, offsetlenght=self.offsetLenght,
                                    plateHeight=self.plateHeight ,platePositionX=self.platePositionX,
                                    platePositionY =self.platePositionY, platePositionZ=self.platePositionZ, extraoffset =self.extraoffset, 
                                    stepRotation= self.stepRotation,offsetlenght2=self.offsetLenght2, servoPlaning = self.servoPlaning,
                                    enLightPos= enLightPos)

        self.ser.clearSerialData()


    def getXYZAndWrite(self,data):
        '''param data : [[position (list), wall (string), pred (int), orentation(3x3 matrix)],...] '''

        if self.modeFixData == False:
            # data = position wall predict orentation
            data = self.package.make10PathLine(dataList= data) 
            # data = [position wall valve orentation]

            data= self.package.sortBestPosition(dataList= data,initial_position=self.initial_position  ,final_position=self.final_position )
            # data = position wall valve orentation

        # data = position wall valve orentation
        for position,wall,valve,orentation in data:
            print('------------\nget position : '+str(position))
            x,y,z = position
            
            R_e = orentation

            dd = self.MAN.d6*R_e[:,2]
            dx = x-dd[0]
            dy = y-dd[1]
            dz = z-dd[2]

            dis = sqrt(pow(x,2)+pow(y-85,2)+pow(z-500,2) )
            try :
                ans = self.MAN.inverse_kinamatic2(dx,dy,dz,self.MAN.DH_param,R_e)
                ans = self.MAN.setJointLimits(ans,self.MAN.jointLimit)

            except :
                sys.exit('IK calculate fail')
        
            if len(ans) >= 1:
                ans = ans[0]
                if self.half_IK:
                    ans = ans[:3]+[0/180*pi,0/180*pi,0/180*pi]

                self.getSetQAndWrite(ans,valve)

        
    def getSetQAndWrite(self,set_q,valve):
        '''param set_Q : [q1,q2,q3,q4,q5,q6]\n\t
             valve : 1 -> open valve, 0 -> closed valve '''
        
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

        new_set_q = copy.deepcopy(set_q)
        new_set_q = self.offsetBackLash(new_set_q)

        set_q_after_offset = [sum(q) for q in zip( [qi[0]*qi[1] for qi in zip(new_set_q,self.gainQ)] ,[radians(qi) for qi in self.offsetQ] )]

        H,Hi = self.MAN.forward_kin(self.MAN.DH_param,set_q)

        H_a = [Hi[:,:4,3]]
        x,y,z = [int(i) for i in  Hi[-1,0:3,3]]
        self.MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])

        print('\nposition per joint : ') 
        for index_Hi in range(Hi.shape[0]):
            xi, yi ,zi = [ int(i) for i in Hi[index_Hi,0:3,3]]
            print('joint '+str(index_Hi+1) + ' : ' + str((xi,yi,zi)))
        

        print('\nFK position : ' +str((x,y,z)))
        print('set_q unoffset : '+str([qi for qi in set_q]))
        # print('set_q unoffset : '+str([degrees(qi) for qi in set_q]))
        # print('set_q backlash : '+str([qi/pi for qi in new_set_q]))
        # print('set_q backlash : '+ str( [degrees(qi) for qi in new_set_q]))

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
        caseBacklash = [self.caseBacklash[index](dataToFunction[index]) for index in range(6) ]

        output = set_q
        for indexQ in range(len(set_q)):
            if set_q[indexQ] > radians(caseBacklash[indexQ]):
                output[indexQ] = set_q[indexQ]-radians(self.offsetBacklash[indexQ]) 
            else :
                output[indexQ] = set_q[indexQ]+radians(self.offsetBacklash[indexQ])
        return output

# a = sendSerial(port=3)
# a.getXYZAndWrite([ [[500,500,800],1,'F'] ])
# a.getSetQAndWrite([95/180*pi,115/180*pi,0/180*pi,135/180*pi,135/180*pi,135/180*pi],0)