__author__ = 'Chaiyaporn Boonyasathian'
__version__ = 1.0
__description__ = 'class to call and predict using Random Forest model'


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as IP
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import joblib

class PaZum():
    def __init__(self, model_path):
        ''' 
        Parameter       model_path  :   Path to saved model    
        Example : 'C:\\Users\cha45\PycharmProjects\AluxandraDrinTheDodge\module\Random_Forest_best_run.sav'
        '''
        try:
            with open(model_path, 'rb') as fo:
                self.model=joblib.load(fo)
            # file = open(model_path, 'rb')
            # self.model = pickle.loads(file.read())
            # file.close()
        except:
            raise FileNotFoundError('There is no file in that path')
            warnings.warn("There is no file in that path, So model will be create according to default value")
            self.model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=2, random_state=None)
        for k, v in self.model.get_params(deep=False).items():
            setattr(self, k, v)

    def predict(self, input, out='prob'):
        ''' 
            Parameter   input  :   list of imageFeature use to predict   
                     ********** Caution ************   input feature should be equivalent of feature use when fit the model
            Example :  come in form of [x0,x1,x2] or [[x0,x1,x2],[x0_0,x0_1,x0_2]]  
            Parameter   out  :   specify output type default is 'prob'  
            Example :  string 'class' or 'prob'
            Return  :   if out is prob and input is a list of shape [n_feature]
                        output will be in shape of [probability of n_class]
                        if out is prob and input is a list of shaprobability of n_classpe [n_samples,n_feature]
                        output will be [n_sample,probability of n_class]
            *** to check the arrangement of class use self.model.classes_
            Usage Example:
                            model = PaZum(Path)
                            input =[0,2,0,1]
                            output = model.predict(input)
        '''
        all_hist = IP.getHis(input)
        all_hog_perfile = IP.getHog(input)
        all_feature = list(map(lambda x, y: np.concatenate([x, y]).tolist(), all_hist, all_hog_perfile))

        # if type(input) != list and type(input) != np.ndarray:
        #     raise TypeError("input Type isn't list or numpy.ndarray")
        # if type(input) != list and type(input) != np.ndarray:
        #     feature = [input]
        # else:
        feature = all_feature
        if out == 'prob':
            pred = self.model.predict_proba(feature)
        else:
            pred = self.model.predict(feature)
        return pred
