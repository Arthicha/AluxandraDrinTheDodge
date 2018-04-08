__author__ = 'Penpat Ritprapa'
__version__ = 1.0
__description__ = 'Store call matlab and call simulink function'

# you need to install matlab.engine with command line from this link
# >> https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html <<



'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import matlab.engine

'''*************************************************
*                                                  *
*                 random function                  *
*                                                  *
*************************************************'''


y = {'arg1': 3, 'arg2': 5}
eng = matlab.engine.start_matlab()
eng.sim("vdp")

def callMatFunc(funcname,argumentDict,outputs):
    eng = matlab.engine.start_matlab()
    a = getattr(eng,str(funcname))(argumentDict,nargout=outputs)
    return a

# callMatFunc("yourfunc",y,1)

def callSimulink(simulinkName):
    eng = matlab.engine.start_matlab()
    # eng.sim(str(simulinkName))
    eng.sim("vdp")
