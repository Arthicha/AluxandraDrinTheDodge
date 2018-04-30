__author__ = 'Penpat Ritprapa'
__version__ = 1.0
__description__ = 'Store call matlab and call simulink function'

# you need to install matlab.engine with command line from this link :
# >> https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html <<
# reference for call simulink  :
# >> https://www.mathworks.com/help/simulink/slref/toworkspace.html <<

'''
    :param funcname[str] : the name of the matlab function your going to call 
    :param argumentDict[dict] : arguments of the function in dictionary type [ex.  y = {'arg1': 3, 'arg2': 5} ] 
    :param output[int] : number of the returned output
    :param simulinkName[str] : the name of the simulink model that you are goin to take the output 
    :param blockName[str] : the name of the block you want to return the output 
    
'''

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import os
import matlab.engine

'''*************************************************
*                                                  *
*                 random function                  *
*                                                  *
*************************************************'''
class MATLAPUTOPPU:
    def __init__(self,path =os.getcwd()+os.sep+'module'+os.sep+'matlab'):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(path)

    def callMatFunc(self,funcname,argumentDict,outputs):
        if str(type(argumentDict)) in ["<class 'mlarray.int16'>","<class 'dict'>"]:
            argument = argumentDict
        elif str(type(argumentDict)) in ["<class 'list'>","<class 'tuple'>"]:
            argument = matlab.int16(argumentDict)
        else :
            argument = argumentDict

        getRes = getattr(self.eng,str(funcname))(argument,nargout=outputs)
        return getRes

    def callSimulink(self,simulinkName,blockName):

        Sim = self.eng.sim(simulinkName,'SimulationMode','normal')
        getSim = Sim.get(blockName)
        return getSim

    def stopMatlab(self):
        self.eng.qut()

# send array to matlab function and return
# eng = matlab.engine.start_matlab()
# A = matlab.int8([1,2,3,4,5])
# y = {'arg1': A , 'arg2': A}
# r = callMatFunc('yourfunc',y,1)

# a = matlab.int16([[0],[0],[0],[0],[0],[0]])
# box,laser = callMatFunc('collision_check',a,1)[0]
# print((box,laser))
