import serial
import time
from math import degrees, radians

class serial_commu():


    def __init__(self,port=3,baud=9600,t=0.01,sendSerial=True, manualStep = False):
        self.sendSerial = sendSerial

        self.ser = serial.Serial()
        self.ser.baudrate = baud
        self.ser.timeout = t
        self.ser.rts = True
        self.ser.dtr = True
        time.sleep(1)
        self.ser.rts = False
        self.ser.dtr = False
        self.ser.port = 'COM'+str(port)
        if self.ser.is_open:
            self.ser.close()
        self.ser.open()
        self.ser.flush()
        self.ser.close()
        self.ser.open()

        self.manualStep = manualStep
        # while self.ser.in_waiting:
        #     self.ser.read()


    def write(self,q=[0,0,0,0,0,0],jointLimit= [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]] ,offset= [0,0,0,0,0,0],valve=0,gainQ=[1,1,1,1,1,1]):
        new_q = [q[0]*gainQ[0],q[1]*gainQ[1],q[2]*gainQ[2],q[3]*gainQ[3],q[4]*gainQ[4],q[5]*gainQ[5]]
        string = [int(degrees(sum(qi))) for qi in zip(new_q,[0 for jL in jointLimit],[radians(ofs) for ofs in offset])]+[valve]
        print('serail package : ',self.makeDataWord(string=string))
        if self.manualStep :
                input('\npress any key and enter to try to send serial :')    
        if self.sendSerial:
            print('send serial:')
            self.ser.write(self.makeDataWord(string=string).encode('ascii'))
        time.sleep(0.1)

    def read(self,length=1):
        word = ''
        while self.ser.in_waiting and len(word) < length:
            word+=str(self.ser.read().decode('ascii'))
        return word

    def readLine(self,length=26):
        
        word = str(self.ser.readline(length+2).decode('ascii'))[:-2]

        return word

    def readAll(self):
        return str(self.ser.read_all().decode('ascii'))
    
    def clearSerialData(self):
        self.ser.read_all()
        return 0

    def makeDataWord(self,string):
        key = 'g'
        key+=(str(string[0]).zfill(3)+','+str(string[1]).zfill(3)+','+str(string[2]).zfill(3)+','
            +str(string[3]).zfill(3)+','+str(string[4]).zfill(3)+','+str(string[5]).zfill(3)+','+str(string[6]))
            
        return key
