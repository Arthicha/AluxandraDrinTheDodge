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

def callMatFunc(funcname,argumentDict,outputs,path =os.getcwd()+os.sep+'matlab'):
    eng = matlab.engine.start_matlab()
    eng.cd(path)
    # eng.foo()
    getRes = getattr(eng,str(funcname))(argumentDict,nargout=outputs)
    return getRes

def callSimulink(simulinkName,blockName,path =os.getcwd()+os.sep+'matlab'):
    eng = matlab.engine.start_matlab()
    eng.cd(path)
    Sim = eng.sim(simulinkName,'SimulationMode','normal')
    getSim = Sim.get(blockName)
    return getSim

# send array to matlab function and return
# eng = matlab.engine.start_matlab()
# A = matlab.int8([1,2,3,4,5])
# y = {'arg1': A , 'arg2': A}
# r = callMatFunc('yourfunc',y,1)

# a = matlab.int16([[0],[0],[0],[0],[0],[0]])
# box,laser = callMatFunc('collision_check',a,1)[0]
# print((box,laser))
