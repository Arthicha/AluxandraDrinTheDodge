import time
import serial
import sys
import math
import numpy as np
import cv2
from pynput import keyboard

from module.serialCommu import serial_commu
from module.MANipulatorKinematics import MANipulator


if __name__ == '__main__':
    ser = serial_commu(port=3)

    d1 = 500.0
    a1 = 85.0
    a2 = 300.0
    a3 = 30.0
    d4 = 448.0
    d6 = 112.0
    
    # mode
    MOD_WORKSPACE = 0
    MOD_INVKIN = 1
    MOD_JACOBIAN = 2

    # mode of operation
    MODE = MOD_INVKIN
    # define DH parameter
    DH_param = np.array([[0,d1,a1,math.pi/2],
                        [0,0,a2,0],
                        [math.pi/2,0,a3,math.pi/2], # fixed
                        [0,d4,0,-math.pi/2],
                        [0,0,0,math.pi/2],
                        [0,d6,0,0]],dtype=np.float32)
    jointLimit = [[-2*math.pi,2*math.pi],
                [0,math.pi],
                [-math.pi,math.pi],
                [-3*math.pi/4,3*math.pi/4],
                [-math.pi/2,math.pi/2],
                [-math.pi/2,math.pi/2]]

    ofset = [0,0,0,0,0,0]
    DELTA = 1

    q = [0,0,0,0,0,0,0]
    MAN = MANipulator()
    RE_F = np.array([[0,0,1],
                    [0,1,0],
                    [1,0,0]])
    RE_L = np.array([[1,0,0],
                    [0,0,-1],
                    [0,1,0]])
    RE_R = np.array([[0,1,0],
                    [0,0,1],
                    [1,0,0]])
    RE_B = np.array([[0,-1,0],
                    [1,0,0],
                    [0,0,-1]])

    ans = [[0,0,0,0,0,0]]
    x = 600
    y = 0
    z = 530
    key = -1
    step = 20
    R_e = RE_F

 
    
    if True:
        while(1) :
            x = 100
            y = 200 
            z = 300
            dd = d6*R_e[:,2]

            dx = x-dd[0]
            dy = y-dd[1]
            dz = z-dd[2]

            try:
                ans = MAN.inverse_kinamatic2(dx,dy,dz,DH_param,R_e)
                #ans=MAN.inverse_angle(dx,dy,dz,R_e,DH_param,BIAS=[0.05,0.05],Target=[x,y,z])
            except:
                sys.exit('inverse fail')
            ans = MAN.setJointLimits(ans,jointLimit,dq = 0.5)
            
            for set_q in ans:

                ser.write(q=set_q, jointLimit=jointLimit, ofset=ofset, valve=0)

                
                print(ser.readLine())
        
        


    else :

        if MODE is MOD_WORKSPACE:
            MAN.plotWorkSpace([0,1,1,0,0,0],DH_param,jointLimit)
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
                    R_e = RE_F
                    print('\tKEYBOARDINPUT:', 'i')
                elif key == 106:
                    R_e = RE_R
                    print('\tKEYBOARDINPUT:', 'j')
                elif key == 107:
                    R_e = RE_B
                    print('\tKEYBOARDINPUT:', 'k')
                elif key == 108:
                    R_e = RE_L
                    print('\tKEYBOARDINPUT:', 'l')


                #x = randint(0,500)
                #y = randint(0,500)
                #z = randint(0,1000)

                dd = d6*R_e[:,2]
                #dd = [0,0,0]
                dx = x-dd[0]
                dy = y-dd[1]
                dz = z-dd[2]
                #q = [1.2,0.21,-0.2]
                #A = MAN.forward2(q[0],q[1],q[2],DH_param)
                #B,_ = MAN.forward_kin(DH_param[:4,:],q+[0.0])
                #print(A)
                #print(B)
                #input('>>>')
                try:
                    ans = MAN.inverse_kinamatic2(dx,dy,dz,DH_param,R_e)
                    #ans=MAN.inverse_angle(dx,dy,dz,R_e,DH_param,BIAS=[0.05,0.05],Target=[x,y,z])
                except:
                    sys.exit('inverse fail')
                ans = MAN.setJointLimits(ans,jointLimit,dq = 0.5)
                for ans_q in ans:
                    H,_ = MAN.forward_kin(DH_param,ans_q)
                    P = H[:3,-1]
                    error = abs(x-P[0])+abs(y-P[1])+abs(z-P[2])
                    #print('error:',error)
                if len(ans) == 0:
                    sys.exit('joint limit fail')
                H_a = []
                for set_q in ans:
                    q1,q2,q3,q4,q5,q6 = set_q

                    #q1,q2,q3,q4 = set_q
                    H,Hi = MAN.forward_kin(DH_param,[q1,q2,q3,q4,q5,q6])
                    #H,Hi = MAN.forward_kin(DH_param[:4,:],[q1,q2,q3,q4])
                    H_a.append(Hi[:,:4,3])

                    ser.write(q=set_q, jointLimit=jointLimit, ofset=ofset, valve=0)
                    print([int(math.degrees(i[0]-i[1])) for i in zip(set_q,[j[0] for j in jointLimit]) ]+[0])
                    print(ser.readLine())
                
                key = MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])
        elif MODE is MOD_JACOBIAN:
            J = MAN.Jackobian([0,0,0,0,0,0],a1=DH_param[0][2],a2=DH_param[1][2],a3=DH_param[1][2],d4=DH_param[3][1],d6=DH_param[5][1])
            print(J)
        


        