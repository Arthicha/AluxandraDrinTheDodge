import math
import sys
import time

import cv2
import numpy as np
import serial

from module.MANipulatorKinematics import MANipulator
from module.serialCommu import serial_commu

if __name__ == '__main__':
    ser = serial_commu(port=3)


    
    # mode
    MOD_WORKSPACE = 0
    MOD_INVKIN = 1
    MOD_JACOBIAN = 2

    # mode of operation
    MODE = MOD_INVKIN
    # define DH parameter

    MAN = MANipulator()
    
    x = 600
    y = 0
    z = 530
    key = -1
    step = 20
    R_e = MAN.RE_F

    if False :
        while(1):
            for x,y,z in zip([100,75,200,600],[50,100,0,0],[80,200,200,530]):

                dd = MAN.d6*R_e[:,2]

                dx = x-dd[0]
                dy = y-dd[1]
                dz = z-dd[2]

                try:
                    ans = MAN.inverse_kinamatic2(dx,dy,dz,MAN.DH_param,R_e)
                    #ans=MAN.inverse_angle(dx,dy,dz,R_e,DH_param,BIAS=[0.05,0.05],Target=[x,y,z])
                except:
                    sys.exit('inverse fail')
                print((x,y,z))
                
                ans = MAN.setJointLimits(ans,MAN.jointLimit)
                
                for set_q in ans:
                    
                    ser.write(q=set_q, jointLimit=MAN.jointLimit, ofset=MAN.ofset, valve=0)

                    time.sleep(1)
                    print(ser.readLine())
                    

    else :

        if MODE is MOD_WORKSPACE:
            MAN.plotWorkSpace([0,1,1,0,0,0],MAN.DH_param,MAN.jointLimit)
        elif MODE is MOD_INVKIN:
            
            
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


                #x = randint(0,500)
                #y = randint(0,500)
                #z = randint(0,1000)

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
