__author__ = 'Zumo Arthicha Srisuchinnawong'
__version__ = 2.0
__description__ = 'class of manipulator'

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from random import randint
import sys
from math import *


'''
description: this is class of manipulator, in side this class there are 4 method that are commonly use,
forward_kin - a forward kinematics function, plotWorkSpace - visualization of the manipulator work space,
inverse_kinematics2 - a method that compute inverse kinematics of the manipulator and setJointLimit - a
method that eliminate answer that out of joint limit.
example: you can find the example program at the bottom section of this file.
'''


class MANipulator():   

    def __init__(self):
        self.d1 = 500.0
        self.a1 = 85.0
        self.a2 = 300.0
        self.a3 = 30.0
        self.d4 = 448.0
        self.d6 = 112.0
        self.DH_param = np.array([[0,self.d1,self.a1,math.pi/2],
                            [0,0,self.a2,0],
                            [math.pi/2,0,self.a3,math.pi/2], # fixed
                            [0,self.d4,0,-math.pi/2],
                            [0,0,0,math.pi/2],
                            [0,self.d6,0,0]],dtype=np.float32)
        self.jointLimit = [[-1/4*math.pi,5/4*math.pi],
                    [1/12*math.pi,math.pi],
                    [-11/12*math.pi,0*math.pi],
                    [-3/4*math.pi,3/4*math.pi],
                    [-3/4*math.pi,3/4*math.pi],
                    [-3/4*math.pi,3/4*math.pi]]



        self.q = [0,0,0,0,0,0,0]
        self.RE_F = np.array([[0,1,0],
                    [0,0,1],
                    [1,0,0]],dtype= np.float)
        self.RE_L = np.array([[0,0,-1],
                        [0,1,0],
                        [1,0,0]],dtype= np.float)
        self.RE_R = np.array([[0,0,1],
                        [0,1,0],
                        [1,0,0]],dtype= np.float)
        self.RE_B = np.array([[0,-1,0],
                        [1,0,0],
                        [0,0,-1]],dtype= np.float)



    def RotTranz(self,DH,q):

        rotz = np.array([[cos(q+DH[0]),-sin(q+DH[0]),0,0],
                         [sin(q+DH[0]),cos(q+DH[0]),0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
        transz = np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,DH[1]],
                           [0,0,0,1]])
        tranx = np.array([[1,0,0,DH[2]],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
        rotx = np.array([[1,0,0,0],
                         [0,cos(DH[3]),-sin(DH[3]),0],
                         [0,sin(DH[3]),cos(DH[3]),0],
                         [0,0,0,1]])
        H = np.matmul(np.matmul(np.matmul(rotz,transz),tranx),rotx)
        return H


    def forward_kin(self,DH_parameter,q):
        H = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        Hi = []
        for i in range(0,len(DH_parameter)):
            #print(q[i],DH_parameter[i])
            H = np.matmul(H,self.RotTranz(DH_parameter[i],q[i]))
            Hi.append(H)
        return (H,np.array(Hi))

    def plot(self,orientation,matplotLibs=False,plotTarget=True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0,len(orientation)):
            listPos = orientation[i]
            init = [0,0,0]
            link_color = ['k-','m-','b-','c-']
            for pos in listPos:
                ax.plot([init[0],pos[0]],[init[1],pos[1]],[init[2],pos[2]],link_color[i],linewidth=3.0)
                init = [pos[0],pos[1],pos[2]]
            if plotTarget != None:

                ax.plot([0,plotTarget[0]],[0,0],[0,0],'r--')
                ax.plot([plotTarget[0],plotTarget[0]],[0,plotTarget[1]],[0,0],'g--')
                ax.plot([plotTarget[0],plotTarget[0]],[plotTarget[1],plotTarget[1]],[0,plotTarget[2]],'b--')
        ax.set_xlim3d(-500,500)
        ax.set_ylim3d(-250-85, 1000-250-85)
        ax.set_zlim3d(0,1000)
        ax.set_xlabel('x_0-direction')
        ax.set_ylabel('y_0-direction')
        #fig.canvas.draw()
        if matplotLibs:
            plt.show(fig)
            return None
        else:
            fig.savefig('inverse.png')
            img = cv2.imread('inverse.png')
            cv2.imshow('inverse kinematics',img)
            key = cv2.waitKey(3)
        return key

    def plotWorkSpace(self,ena_q,DH_param,q_lim,step=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-500,500)
        ax.set_ylim3d(-250-85, 1000-250-85)
        ax.set_zlim3d(0,1000)
        ax.set_xlabel('x_0-direction')
        ax.set_ylabel('y_0-direction')
        rang = []
        firsty = [False,0]
        for i in range(0,len(ena_q)):
            if ena_q[i]:
                if firsty[0] is False:
                    firsty[0] = True
                    firsty[1] = i
                rang.append([int(q_lim[i][0]*180.0//math.pi),int(q_lim[i][1]*180.0//math.pi)])
            else:
                if i is 0:
                    rang.append([0,180])
                else:
                    rang.append([0,1])
        if ena_q[0]:
            list1 = range(rang[0][0],rang[0][1],step)
        else:
            list1 = rang[0]
        for qq1 in list1:
            print('q1:',qq1)
            q1 = qq1*math.pi/180.0
            for qq2 in range(rang[1][0],rang[1][1],step):
                if firsty[1] is 1:
                    print(qq2*100.0/rang[1][1])
                q2 = qq2*math.pi/180.0
                for qq3 in range(rang[2][0],rang[2][1],step):
                    if firsty[1] is 3:
                        print(qq3*100.0/rang[2][1])
                    q3 = qq3*math.pi/180.0
                    for qq4 in range(rang[3][0],rang[3][1],step):
                        if firsty[1] is 4:
                            print(qq4*100.0/rang[3][1])
                        q4 = qq4*math.pi/180.0
                        for qq5 in range(rang[4][0],rang[4][1],step):
                            if firsty[1] is 5:
                                print(qq5*100.0/rang[3][1])
                            q5 = qq5*math.pi/180.0
                            for qq6 in range(rang[5][0],rang[5][1],step):
                                q6 = qq6*math.pi/180.0
                                H,Hi = MAN.forward_kin(DH_param,[q1,q2,q3,q4,q5,q6])
                                P = H[0:3,3]
                                ax.plot([P[0]],[P[1]],[P[2]],'ro')
        plt.show()

    def setJointLimits(self,q_list,q_lim,dq=0.0):
        ans = []
        for q_set in q_list:
            check = True
            for i in range(0,len(q_set)):
                if (q_set[i] > q_lim[i][1]+dq) or (q_set[i] < q_lim[i][0]-dq):
                    check = False
                    break
            if check:
                ans.append(q_set)
        return ans

    def Jackobian(self,q,a1=0,a2=0,a3=0,d4=0,d6=0):


        q1,q2,q3,q4,q5,q6 = q
        J = [[a3*cos(q2)*sin(q1)*sin(q3)-a2*cos(q2)*sin(q1)-d4*cos(q2)*cos(q3)*sin(q1) - a1*sin(q1) + a3*cos(q3)*sin(q1)*sin(q2) + d6*cos(q1)*sin(q4)*sin(q5) + d4*sin(q1)*sin(q2)*sin(q3) - d6*cos(q2)*cos(q3)*cos(q5)*sin(q1) + d6*cos(q5)*sin(q1)*sin(q2)*sin(q3) + d6*cos(q2)*cos(q4)*sin(q1)*sin(q3)*sin(q5) + d6*cos(q3)*cos(q4)*sin(q1)*sin(q2)*sin(q5) , -cos(q1)*(a2*sin(q2) + a3*cos(q2)*cos(q3) + d4*cos(q2)*sin(q3) + d4*cos(q3)*sin(q2) - a3*sin(q2)*sin(q3) + d6*cos(q2)*cos(q5)*sin(q3) + d6*cos(q3)*cos(q5)*sin(q2) + d6*cos(q2)*cos(q3)*cos(q4)*sin(q5) - d6*cos(q4)*sin(q2)*sin(q3)*sin(q5)), -cos(q1)*(a3*cos(q2)*cos(q3) + d4*cos(q2)*sin(q3) + d4*cos(q3)*sin(q2) - a3*sin(q2)*sin(q3) + d6*cos(q2)*cos(q5)*sin(q3) + d6*cos(q3)*cos(q5)*sin(q2) + d6*cos(q2)*cos(q3)*cos(q4)*sin(q5) - d6*cos(q4)*sin(q2)*sin(q3)*sin(q5)), d6*sin(q5)*(cos(q4)*sin(q1) + cos(q1)*cos(q2)*sin(q3)*sin(q4) + cos(q1)*cos(q3)*sin(q2)*sin(q4)), d6*cos(q5)*sin(q1)*sin(q4) - d6*cos(q1)*cos(q2)*cos(q3)*sin(q5) + d6*cos(q1)*sin(q2)*sin(q3)*sin(q5) - d6*cos(q1)*cos(q2)*cos(q4)*cos(q5)*sin(q3) - d6*cos(q1)*cos(q3)*cos(q4)*cos(q5)*sin(q2),0],
             [ a1*cos(q1) + a2*cos(q1)*cos(q2) + d4*cos(q1)*cos(q2)*cos(q3) - a3*cos(q1)*cos(q2)*sin(q3) - a3*cos(q1)*cos(q3)*sin(q2) - d4*cos(q1)*sin(q2)*sin(q3) + d6*sin(q1)*sin(q4)*sin(q5) + d6*cos(q1)*cos(q2)*cos(q3)*cos(q5) - d6*cos(q1)*cos(q5)*sin(q2)*sin(q3) - d6*cos(q1)*cos(q2)*cos(q4)*sin(q3)*sin(q5) - d6*cos(q1)*cos(q3)*cos(q4)*sin(q2)*sin(q5), -sin(q1)*(a2*sin(q2) + a3*cos(q2)*cos(q3) + d4*cos(q2)*sin(q3) + d4*cos(q3)*sin(q2) - a3*sin(q2)*sin(q3) + d6*cos(q2)*cos(q5)*sin(q3) + d6*cos(q3)*cos(q5)*sin(q2) + d6*cos(q2)*cos(q3)*cos(q4)*sin(q5) - d6*cos(q4)*sin(q2)*sin(q3)*sin(q5)), -sin(q1)*(a3*cos(q2)*cos(q3) + d4*cos(q2)*sin(q3) + d4*cos(q3)*sin(q2) - a3*sin(q2)*sin(q3) + d6*cos(q2)*cos(q5)*sin(q3) + d6*cos(q3)*cos(q5)*sin(q2) + d6*cos(q2)*cos(q3)*cos(q4)*sin(q5) - d6*cos(q4)*sin(q2)*sin(q3)*sin(q5)), d6*sin(q5)*(cos(q2)*sin(q1)*sin(q3)*sin(q4) - cos(q1)*cos(q4) + cos(q3)*sin(q1)*sin(q2)*sin(q4)), d6*sin(q1)*sin(q2)*sin(q3)*sin(q5) - d6*cos(q2)*cos(q3)*sin(q1)*sin(q5) - d6*cos(q1)*cos(q5)*sin(q4) - d6*cos(q2)*cos(q4)*cos(q5)*sin(q1)*sin(q3) - d6*cos(q3)*cos(q4)*cos(q5)*sin(q1)*sin(q2),0],
            [0,a2*cos(q2) + d4*cos(q2)*cos(q3) - a3*cos(q2)*sin(q3) - a3*cos(q3)*sin(q2) - d4*sin(q2)*sin(q3) + d6*cos(q2)*cos(q3)*cos(q5) - d6*cos(q5)*sin(q2)*sin(q3) - d6*cos(q2)*cos(q4)*sin(q3)*sin(q5) - d6*cos(q3)*cos(q4)*sin(q2)*sin(q5),            d4*cos(q2)*cos(q3) - a3*cos(q2)*sin(q3) - a3*cos(q3)*sin(q2) - d4*sin(q2)*sin(q3) + d6*cos(q2)*cos(q3)*cos(q5) - d6*cos(q5)*sin(q2)*sin(q3) - d6*cos(q2)*cos(q4)*sin(q3)*sin(q5) - d6*cos(q3)*cos(q4)*sin(q2)*sin(q5),                                                                 -d6*cos(q2 + q3)*sin(q4)*sin(q5),                                                              d6*cos(q2)*cos(q3)*cos(q4)*cos(q5) - d6*cos(q3)*sin(q2)*sin(q5) - d6*cos(q2)*sin(q3)*sin(q5) - d6*cos(q4)*cos(q5)*sin(q2)*sin(q3),                                                                                                                            0],
            [                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                       sin(q1),                                                                                                                                                                                                                          sin(q1),                                                                             cos(q2 + q3)*cos(q1),                                                                                                            cos(q4)*sin(q1) + cos(q1)*cos(q2)*sin(q3)*sin(q4) + cos(q1)*cos(q3)*sin(q2)*sin(q4), cos(q2 + q3)*cos(q1)*cos(q5) - sin(q5)*(cos(q1)*cos(q2)*cos(q4)*sin(q3) - sin(q1)*sin(q4) + cos(q1)*cos(q3)*cos(q4)*sin(q2))],
            [                                                                                                                                                                                                                              0,                                                                                                                                                                                                                                      -cos(q1),                                                                                                                                                                                                                         -cos(q1),                                                                             cos(q2 + q3)*sin(q1),                                                                                                            cos(q2)*sin(q1)*sin(q3)*sin(q4) - cos(q1)*cos(q4) + cos(q3)*sin(q1)*sin(q2)*sin(q4), cos(q2 + q3)*cos(q5)*sin(q1) - sin(q5)*(cos(q1)*sin(q4) + cos(q2)*cos(q4)*sin(q1)*sin(q3) + cos(q3)*cos(q4)*sin(q1)*sin(q2))],
            [                                                                                                                                                                                                                                                                                                                                                    1,                                                                                                                                                                                                                                             0,                                                                                                                                                                                                                                0,                                                                                     sin(q2 + q3),                                                                                                                                                                          -cos(q2 + q3)*sin(q4),sin(q2 + q3)*cos(q5) + cos(q2 + q3)*cos(q4)*sin(q5)]]

        return np.array(J)

    def inverse_kinamatic2(self,dx,dy,dz,DH_param,R_e):
        #print('input',dx,dy,dz)
        d1 = DH_param[0][1]
        d4 = DH_param[3][1]
        d6 = DH_param[5][1]
        a1 = DH_param[0][2]
        a2 = DH_param[1][2]
        a3 = DH_param[2][2]


        q1 = atan2(dy,dx)


        x = sqrt(pow(dx,2)+pow(dy,2)) - a1
        x_sq = pow(dx,2)+pow(dy,2)+pow(a1,2)-2*a1*sqrt(pow(dx,2)+pow(dy,2))
        y = dz - d1
        l1 = a2
        l2 = sqrt(pow(a3,2)+pow(d4,2))
        l2_sq = pow(a3,2)+pow(d4,2)
        gamma = atan2(a3,d4)

        qq3 = []
        c3 =(x_sq+pow(y,2)-pow(l1,2)-l2_sq)/(2*l1*l2)
        s3 = sqrt(1-pow(c3,2))
        for sig in [1.0,-1.0]:
            qq3.append(atan2(sig*s3,c3)-gamma)



        q = []
        q3_ind = 0
        for q3 in qq3:
            L1 = l1+l2*cos(q3+gamma)
            L2 = l2*sin(q3+gamma)
            r = sqrt(pow(L1,2)+pow(L2,2))
            beta = atan2(L2,L1)
            k1 = r*cos(beta)
            k2 = r*sin(beta)
            q2 = atan2(y,x)-atan2(k2,k1)


            H,_ = self.forward_kin(DH_param[:3,:],[q1,q2,q3])
            R = np.matmul(np.transpose(H[:3,:3]),R_e)
            Rz = R[:,-1]
            s5 = sqrt(pow(Rz[0],2)+pow(Rz[1],2))
            c5 = Rz[2]
            q3_ind += 1
            #q5 = asin(sqrt(pow(Rz[0],2)+pow(Rz[1],2)))
            q5 = acos(c5)
            q4 = atan2(Rz[1],Rz[0])
            #print('check',Rz)
            if (q4 > math.pi/2) or (q4 < -math.pi/2):
                #print('in')
                if (q4 > math.pi/2):
                    q4 -= math.pi
                else:
                    q4 += math.pi
                q5 = -q5

            Rx = R[:, 0]
            q6 = atan((sin(q4)*Rx[0]-cos(q4)*Rx[1])*sin(q5)/Rx[2])
            q.append([q1,q2,q3,q4,q5,q6])

        return q

'''
                        example program
position control                        end effector control
        w : x++                                 j : point left
        s : x--                                 k : point down
        d : y++                                 i : point forward
        a : y--                                 l : point right
        e : z++
        q : z--
'''

# if 1:

#     # define constant
#     d1 = 500.0
#     a1 = 85.0
#     a2 = 300.0
#     a3 = 30.0
#     d4 = 448.0
#     d6 = 112.0

#     # mode
#     MOD_WORKSPACE = 0
#     MOD_INVKIN = 1
#     MOD_JACOBIAN = 2

#     # mode of operation
#     MODE = MOD_INVKIN
#     # define DH parameter
#     DH_param = np.array([[0,d1,a1,math.pi/2],
#                          [0,0,a2,0],
#                          [math.pi/2,0,a3,math.pi/2], # fixed
#                          [0,d4,0,-math.pi/2],
#                          [0,0,0,math.pi/2],
#                          [0,d6,0,0]],dtype=np.float32)
#     jointLimit = [[-2*math.pi,2*math.pi],
#                   [0,math.pi],
#                   [-math.pi,math.pi],
#                   [-3*math.pi/4,3*math.pi/4],
#                   [-math.pi/2,math.pi/2],
#                   [-math.pi/2,math.pi/2]]

#     DELTA = 1

#     q = [0,0,0,0,0,0,0]
#     MAN = MANipulator()
#     RE_F = np.array([[0,0,1],
#                      [0,1,0],
#                      [1,0,0]])
#     RE_L = np.array([[1,0,0],
#                      [0,0,-1],
#                      [0,1,0]])
#     RE_R = np.array([[0,1,0],
#                      [0,0,1],
#                      [1,0,0]])
#     RE_B = np.array([[0,-1,0],
#                      [1,0,0],
#                      [0,0,-1]])

#     if MODE is MOD_WORKSPACE:
#         MAN.plotWorkSpace([0,1,1,0,0,0],DH_param,jointLimit)
#     elif MODE is MOD_INVKIN:
#         ans = [[0,0,0,0,0,0]]
#         x = 600
#         y = 0
#         z = 530
#         key = -1
#         step = 20
#         R_e = RE_F
#         while(1):
#             #BIAS = [0.07,0.07]
#             # define position
#             if key == 115:
#                 x -= step
#                 print('\tKEYBOARDINPUT:','s')
#             elif key == 119:
#                 x += step
#                 print('\tKEYBOARDINPUT:', 'w')
#             elif key == 97:
#                 y -= step
#                 print('\tKEYBOARDINPUT:', 'a')
#             elif key == 100:
#                 y += step
#                 print('\tKEYBOARDINPUT:', 'd')
#             elif key == 101:
#                 z += step
#                 print('\tKEYBOARDINPUT:', 'e')
#             elif key == 113:
#                 z -= step
#                 print('\tKEYBOARDINPUT:', 'q')
#             elif key == 105:
#                 R_e = RE_F
#                 print('\tKEYBOARDINPUT:', 'i')
#             elif key == 106:
#                 R_e = RE_R
#                 print('\tKEYBOARDINPUT:', 'j')
#             elif key == 107:
#                 R_e = RE_B
#                 print('\tKEYBOARDINPUT:', 'k')
#             elif key == 108:
#                 R_e = RE_L
#                 print('\tKEYBOARDINPUT:', 'l')


#             #x = randint(0,500)
#             #y = randint(0,500)
#             #z = randint(0,1000)

#             dd = d6*R_e[:,2]
#             #dd = [0,0,0]
#             dx = x-dd[0]
#             dy = y-dd[1]
#             dz = z-dd[2]
#             #q = [1.2,0.21,-0.2]
#             #A = MAN.forward2(q[0],q[1],q[2],DH_param)
#             #B,_ = MAN.forward_kin(DH_param[:4,:],q+[0.0])
#             #print(A)
#             #print(B)

#             #input('>>>')
#             try:
#                 ans = MAN.inverse_kinamatic2(dx,dy,dz,DH_param,R_e)
#                 #ans=MAN.inverse_angle(dx,dy,dz,R_e,DH_param,BIAS=[0.05,0.05],Target=[x,y,z])
#             except:
#                 sys.exit('inverse fail')
#             ans = MAN.setJointLimits(ans,jointLimit,dq = 0.5)
#             for ans_q in ans:
#                 H,_ = MAN.forward_kin(DH_param,ans_q)
#                 P = H[:3,-1]
#                 error = abs(x-P[0])+abs(y-P[1])+abs(z-P[2])
#                 #print('error:',error)
#             if len(ans) == 0:
#                 sys.exit('joint limit fail')
#             H_a = []
#             for set_q in ans:
#                 q1,q2,q3,q4,q5,q6 = set_q
#                 #q1,q2,q3,q4 = set_q
#                 H,Hi = MAN.forward_kin(DH_param,[q1,q2,q3,q4,q5,q6])
#                 #H,Hi = MAN.forward_kin(DH_param[:4,:],[q1,q2,q3,q4])
#                 H_a.append(Hi[:,:4,3])
#             key = MAN.plot(H_a,matplotLibs=False,plotTarget=[x,y,z])
#     elif MODE is MOD_JACOBIAN:
#         J = MAN.Jackobian([0,0,0,0,0,0],a1=DH_param[0][2],a2=DH_param[1][2],a3=DH_param[1][2],d4=DH_param[3][1],d6=DH_param[5][1])
#         print(J)
