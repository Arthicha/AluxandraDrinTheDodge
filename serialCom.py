import sys
import time
from math import degrees

import cv2
import numpy as np
import serial

from module.MANipulatorKinematics import MANipulator
from module.serialCommu import serial_commu
from module.prePackage import prePackage


# matlab = MATLAPUTOPPU()
# a = [[0],[0],[0],[0],[0],[0]]
# box,laser = matlab.callMatFunc('collision_check',a,1)[0]
# print((box,laser))


if __name__ == '__main__':
    
    platePositionX = 600
    platePositionY = [300,100,-100,-300]
    platePositionZ = [700,500,300]
    ofsetLenght = 20
    plateHeight = 50

    checkLaser = False

    workspace = [-400,600,-500,500,0,1000]

    ser = serial_commu(port=3)
    MAN = MANipulator()
    R_e = MAN.RE_F
    package = prePackage(checkLaser=checkLaser)

    ser.clearSerialData()
    if True:
        # data = zip(zip([0,100,75,75,75,600],[0,50,100,100,100,0],[0,80,200,200,200,530]),['F','F','F','L','R','F'],[0,0,0,0,0,0],[0,0,0,0,0,0])
        # data = zip(zip([75,100,150],[100,125,200],[200,250,300]),[0,5,10],[0,0,0],['F','L','R'])
        data = zip(zip([100,150],[125,100],[250,200]),[5,10],[0,0],['L','R'])
        '''arg datalist = [[3D-position, predict_output, degreee of pai, no. of wall],...]'''

        path = package.make10PathLine(data, ofsetlenght=ofsetLenght)
        data= package.sortBestPosition(path,initial_position=[300,300,300],final_position=[300,300,300])

        
        # data = package.addSubPosition(dataList=data,platePositionX=platePositionX,platePositionY=platePositionY 
        #                     ,platePositionZ= platePositionZ, ofsetlenght=ofsetLenght, plateHeight=plateHeight)
        # print(data)
        for position,wall,valve,ang in data:
            x,y,z = position
            # wall 'F'front,'L'left,'R'right,'B'buttom
            if wall == 'F':
                R_e = MAN.RE_F
            elif wall == 'L':
                R_e = MAN.RE_R
            elif wall == 'R':
                R_e = MAN.RE_L
            elif wall == 'B':
                R_e = MAN.RE_B

            dd = MAN.d6*R_e[:,2]
            dx = x-dd[0]
            dy = y-dd[1]
            dz = z-dd[2]

            try :
                ans = MAN.inverse_kinamatic2(dx,dy,dz,MAN.DH_param,R_e)
                ans = MAN.setJointLimits(ans,MAN.jointLimit)
            except :
                ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                sys.exit('IK calculate fail')
            print((x,y,z))
            H_a = []

            for set_q in ans:
                boxHit,laserHit = package.boxbreak(set_q[:-1]+[ang])

                if boxHit ==1:
                    ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                    sys.exit('BOX HIT!!!')
                
                if laserHit == 1 :
                    if  checkLaser:
                        ser.write(q=[0,0,0,0,0,0], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                        sys.exit('ALERT LASER!!!')
                    else :
                        print('ALERT LASER!!!')
                

                ser.write(q=set_q[:-1]+[ang], jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=valve)
                # time.sleep(1)
                print(ser.readLine(26))
                H,Hi = MAN.forward_kin(MAN.DH_param,set_q[:-1]+[ang])
                H_a.append(Hi[:,:4,3])
                MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])
                
                while ser.readLine(2) == 'ok':
                    # ser.clearSerialData()
                    pass
                
            
    else :
        # mode
        MOD_WORKSPACE = 0
        MOD_INVKIN = 1
        MOD_JACOBIAN = 2

        # mode of operation
        MODE = MOD_INVKIN
        if MODE is MOD_WORKSPACE:
            MAN.plotWorkSpace([0,1,1,0,0,0],MAN.DH_param,MAN.jointLimit)
        elif MODE is MOD_INVKIN:
            
            x = 600
            y = 0
            z = 530
            key = -1
            step = 20
            while(1):
                #BIAS = [0.07,0.07]
                # define position
                if key == 115:
                    x -= step
                    print('\tKEYBOARDINPUT:','s')
                elif key == 119:
                    x += step
                    print('\tKEYBOARDINPUT:', 'w')
                elif key == 97:
                    y -= step
                    print('\tKEYBOARDINPUT:', 'a')
                elif key == 100:
                    y += step
                    print('\tKEYBOARDINPUT:', 'd')
                elif key == 101:
                    z += step
                    print('\tKEYBOARDINPUT:', 'e')
                elif key == 113:
                    z -= step
                    print('\tKEYBOARDINPUT:', 'q')
                elif key == 105:
                    R_e = MAN.RE_F
                    print('\tKEYBOARDINPUT:', 'i')
                elif key == 106:
                    R_e = MAN.RE_R
                    print('\tKEYBOARDINPUT:', 'j')
                elif key == 107:
                    R_e = MAN.RE_B
                    print('\tKEYBOARDINPUT:', 'k')
                elif key == 108:
                    R_e = MAN.RE_L
                    print('\tKEYBOARDINPUT:', 'l')

                dd = MAN.d6*R_e[:,2]

                dx = x-dd[0]
                dy = y-dd[1]
                dz = z-dd[2]

                try:
                    ans = MAN.inverse_kinamatic2(dx,dy,dz,MAN.DH_param,R_e)
                    
                except:
                    sys.exit('inverse fail')
                ans = MAN.setJointLimits(ans,MAN.jointLimit,dq = 0.5)
                for ans_q in ans:
                    H,_ = MAN.forward_kin(MAN.DH_param,ans_q)
                    P = H[:3,-1]
                    error = abs(x-P[0])+abs(y-P[1])+abs(z-P[2])
                    #print('error:',error)
                if len(ans) == 0:
                    sys.exit('joint limit fail')
                H_a = []
                for set_q in ans:
                    q1,q2,q3,q4,q5,q6 = set_q

                    #q1,q2,q3,q4 = set_q
                    H,Hi = MAN.forward_kin(MAN.DH_param,[q1,q2,q3,q4,q5,q6])
                    #H,Hi = MAN.forward_kin(DH_param[:4,:],[q1,q2,q3,q4])
                    H_a.append(Hi[:,:4,3])

                    ser.write(q=set_q, jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)
                    # time.sleep(1)
                    print(ser.readLine())
                
                key = MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])
        elif MODE is MOD_JACOBIAN:
            J = MAN.Jackobian([0,0,0,0,0,0],a1=MAN.DH_param[0][2],a2=MAN.DH_param[1][2],a3=MAN.DH_param[1][2],d4=MAN.DH_param[3][1],d6=MAN.DH_param[5][1])
            print(J)
