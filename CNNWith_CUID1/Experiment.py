import numpy as np
class Experiment():
    # __nusr=[]
    # _nexp=[]
    #exp_data=np.vstack((act_id[exp_idx], act_be[exp_idx], act_en[exp_idx]))
    def __init__(self,nexp,nusr, exp_data,pathDS):
        self._nexp=nexp
        self._nusr=nusr
        self._exp_data=exp_data
        self._X,self._Y,self._Z=self.ReadExperiment(pathDS+ '/acc_exp'+ nexp+ '_user'+ nusr+ '.txt')
        self._GX,self._GY,self._GZ=self.ReadExperiment(pathDS+'/gyro_exp'+ nexp+ '_user'+ nusr+ '.txt')
        # self.__X=X
        # self._Y=Y
        # self.__Z=Z
    
    def get_nexp(self):
        return self._nexp
    def ReadExperiment(self,pathstring):
        fid = open(pathstring,'r')
        x,y,z=np.loadtxt(fname=fid,delimiter=' ',dtype={'names':('x','y','z'),'formats':('f4','f4','f4')},unpack=True)
        fid.close()
        return x,y,z
