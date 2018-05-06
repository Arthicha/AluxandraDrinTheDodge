import sys
import time
from math import degrees, radians, pi, sqrt, pow

import cv2
import numpy as np
import serial

from module.MANipulatorKinematics import MANipulator
from module.serialCommu import serial_commu
from module.prePackage import prePackage

class sendSerial:

    def __init__(self,port=4,checkLaser = False, runMatlab= True, sendSerial= True,
                 pathPlaning = True, initial_position = [200,200,200], recieveSerial= True ,half_IK=False,
                 platePositionX= 600, platePositionY = [300,100,-100,-300], platePositionZ = [700,500,300], extraOfset = 60,
                 ofsetLenght = 20, plateHeight = 50, workspace = [-400,600,-500,500,0,1000], ofsetQ = [205,35,150,0,0,0],
                gainQ = [-1,1,1,1,1,1],modeFixData = False, stepRotation = 5,ofsetLenght2 = 40, servoPlaning = True):
    
        self.platePositionX = platePositionX
        self.platePositionY = platePositionY
        self.platePositionZ = platePositionZ
        self.ofsetLenght = ofsetLenght
        self.ofsetLenght2 = ofsetLenght2
        self.extraOfset = extraOfset
        self.plateHeight = plateHeight

        self.stepRotation = stepRotation

        self.checkLaser = checkLaser

        self.workspace = workspace

        self.ofsetQ = ofsetQ
        self.gainQ = gainQ

        self.runMatLab = runMatlab

        self.sendSerial = sendSerial

        self.pathPlaning = pathPlaning
        self.servoPlaning = servoPlaning

        self.recieveSerial = recieveSerial

        self.half_IK = half_IK

        self.initial_position = initial_position
        self.final_position = self.initial_position
        self.modeFixData = modeFixData

        '''-----------------------------------------------------------------------------'''

        self.ser = serial_commu(port=port, sendSerial=self.sendSerial)
        input('press reset board')
        self.MAN = MANipulator()
        # self.R_e = MAN.RE_R
        self.package = prePackage(pathPlaning=self.pathPlaning, runMatLab=self.runMatLab, ofsetlenght=self.ofsetLenght,
                                    plateHeight=self.plateHeight ,platePositionX=self.platePositionX,
                                    platePositionY =self.platePositionY, platePositionZ=self.platePositionZ, extraOfset =self.extraOfset, 
                                    stepRotation= self.stepRotation,ofsetlenght2=self.ofsetLenght2, servoPlaning = self.servoPlaning)

        self.ser.clearSerialData()


    def getXYZAndWrite(self,data):
        '''param data : [[position (list), wall (string), pred (int), orentation(3x3 matrix)],...] '''

        if self.modeFixData == False:
            # data = position wall predict orentation
            data = self.package.make10PathLine(dataList= data) 
            # data = [position wall valve orentation]
            # print(data)
            data= self.package.sortBestPosition(dataList= data,initial_position=self.initial_position  ,final_position=self.final_position )
            # data = position wall valve orentation

        # data = position wall valve orentation
        for position,wall,valve,orentation in data:
            print('get position : '+str(position))
            x,y,z = position
            
            R_e = orentation

            # # wall 'F'front,'L'left,'R'right,'B'buttom
            # if wall == 'F':
            #     R_e = self.MAN.RE_F
            # elif wall == 'L':
            #     R_e = self.MAN.RE_L
            # elif wall == 'R':
            #     R_e = self.MAN.RE_R
            # elif wall == 'B':
            #     R_e = self.MAN.RE_B

            dd = self.MAN.d6*R_e[:,2]
            dx = x-dd[0]
            dy = y-dd[1]
            dz = z-dd[2]
            dis = sqrt(pow(x,2)+pow(y-85,2)+pow(z-500,2) )
            try :
                ans = self.MAN.inverse_kinamatic2(dx,dy,dz,self.MAN.DH_param,R_e)
                ans = self.MAN.setJointLimits(ans,self.MAN.jointLimit)
            except :
                # ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                sys.exit('IK calculate fail')
        
            if len(ans) >= 1:
                ans = ans[0]
                if self.half_IK:
                    ans = ans[:3]+[0/180*pi,0/180*pi,0/180*pi]

                self.getSetQAndWrite(ans,valve)

                # # display simulator
                # H,Hi = self.MAN.forward_kin(self.MAN.DH_param,ans)
                # H_a = [Hi[:,:4,3]]
                # self.MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])
            # return [ans,valve]

        
    def getSetQAndWrite(self,set_q,valve):
        '''param set_Q : [q1,q2,q3,q4,q5,q6]\n\t
             valve : 1 -> open valve, 0 -> closed valve '''
        
        if self.runMatLab :
            boxHit,laserHit = self.package.boxbreak(set_q)
            if boxHit ==0:
                # ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                sys.exit('BOX HIT!!!')
            
            if laserHit == 1 :
                if  self.checkLaser:
                    # ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                    sys.exit('ALERT LASER!!!')
                else :
                    print('ALERT LASER!!!')

        set_q_after_ofset = [sum(q) for q in zip( [qi[0]*qi[1] for qi in zip(set_q,self.gainQ)] ,[radians(qi) for qi in self.ofsetQ] )]
        # print('ofset_q : '+str([degrees(qi) for qi in set_q_after_ofset]))
        # print('ofset_q : '+str([qi/pi for qi in set_q_after_ofset]))
        H,Hi = self.MAN.forward_kin(self.MAN.DH_param,set_q)
        H_a = [Hi[:,:4,3]]
        x,y,z = [int(i) for i in  Hi[-1,0:3,3]]
        print('FK position : ' +str((x,y,z)))
        print('set_q unofset : '+str([qi/pi for qi in set_q]))
        self.MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])

        self.ser.write(q=set_q_after_ofset,valve=valve)
        # time.sleep(1)
        # print(self.ser.readLine(26))
        
        if self.recieveSerial:
            serRead = self.ser.read() 
            while serRead != 'A' :
                serRead = self.ser.read() 
                # print(serRead)
                time.sleep(0.1)
                # pass
            input('press any key')
        return 0


# a = sendSerial(port=3)
# a.getXYZAndWrite([ [[500,500,800],1,'F'] ])
# a.getSetQAndWrite([95/180*pi,115/180*pi,0/180*pi,135/180*pi,135/180*pi,135/180*pi],0)