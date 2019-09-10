from Experiment import Experiment
import shelve
import numpy as np
from StructData import StructData
import os

class DataSet:
    # training data
    _experiments = []
    _status = []
    _test_user_ids=[2, 4, 9, 10, 12, 13, 18, 20, 24]

    def __init__(self,pathDS, status='l'):
        self.__pathDS = pathDS
        self._status = status
        filename = 'ShelveExperiments.out'
        if(status == 'l'):
            try:
                my_shelve = shelve.open(filename)
                self._experiments = my_shelve['_experiments']
            except :
                print("cannot open shelve experiments")
            finally:
                my_shelve.close()

        elif (status == 's'):
            self.ExtractAndSaveDSasExperiments(filename)

    def ReadLabels(self):
        fd = open(file=self.__pathDS+'/labels.txt', mode='r')
        exp_id, usr_id, act_id, act_be, act_en = np.loadtxt(fname=fd, delimiter=' ', dtype={'names': (
            'exp_id', 'usr_id', 'act_id', 'act_begin', 'act_end'), 'formats': ('i4', 'i4', 'i4', 'i4', 'i4')}, unpack=True)
        fd.close()
        return exp_id, usr_id, act_id, act_be, act_en

    def ReadAllExperiments(self, exp_id, usr_id, act_id, act_be, act_en):
        idx_files = np.vstack((exp_id, usr_id))
        idx_files = idx_files.T[np.newaxis][0]
        idx_files = np.unique(idx_files, axis=0)
        for i in range(idx_files.shape[0]):
            print('exp '+str(i)+ ' from '+ str(idx_files.shape[0]))
            nexp = np.array2string(idx_files[i, 0])
            if(idx_files[i, 0] < 10):
                nexp = '0' + nexp
            nusr = np.array2string(idx_files[i, 1])
            if(idx_files[i, 1] < 10):
                nusr = '0' + nusr

            # extract all data (usr,action begining, and action end in one Experiment or file--each file or Experiment has one user but different activities
            exp_idx = np.nonzero(exp_id == idx_files[i, 0])
            exp_data = np.vstack(
                (act_id[exp_idx], act_be[exp_idx], act_en[exp_idx]))
            exp_data = exp_data.T[np.newaxis][0]

            exper = Experiment(nexp=nexp, nusr=nusr,
                               exp_data=exp_data, pathDS=self.__pathDS)
            self._experiments.append(exper)

    def SplitTrainTest(self, test_user_ids,_experiments):
        _trainingExperiments = []
        _testExperiments = []
        _test_user_ids = test_user_ids
        for expr in _experiments:
            if not int(expr._nusr) in test_user_ids:  # check if userId is in test or not
                _trainingExperiments.append(expr)
            else:
                _testExperiments.append(expr)
        return _trainingExperiments, _testExperiments

    def SplitExprIntoSegmentSize(self,exper,segment_size):
        paramX = []
        paramY = []
        paramZ = []
        paramGX = []
        paramGY = []
        paramGZ = []
        features = []
        answers_raw = []
        for j in range(exper._exp_data.shape[0]):
            if exper._exp_data[j, 0] < 7:  # activites ids from 1 to 6 and others are out
                k = exper._exp_data[j, 1] - 1
                while k + segment_size <= exper._exp_data[j, 2] - 1:
                    x_add = exper._X[k: k + segment_size: 1]
                    y_add = exper._Y[k: k + segment_size: 1]
                    z_add = exper._Z[k: k + segment_size: 1]

                    paramX = np.hstack((paramX, x_add))
                    paramY = np.hstack((paramY, y_add))
                    paramZ = np.hstack((paramZ, z_add))

                    ff = self.Extract_Features(x_add, y_add, z_add)
                    features.append(ff)

                    paramGX = np.hstack(
                        (paramGX, exper._GX[k:k+segment_size:1]))
                    paramGY = np.hstack(
                        (paramGY, exper._GY[k:k+segment_size:1]))
                    paramGZ = np.hstack(
                        (paramGZ, exper._GZ[k:k+segment_size:1]))

                    answers_raw = np.hstack((answers_raw, exper._exp_data[j, 0]))
                    k = k + segment_size//2
        return StructData(paramX, paramY, paramZ, paramGX, paramGY, paramGZ, features, answers_raw)

    def GetAllExperiments(self):
        return self._experiments

    def GetExperimentWithID(self, Id):
        return self._experiments[Id-1]

    def Extract_Features(self,xs, ys, zs):

        n = len(xs)  # x.shape[0]
        feature_vector = np.hstack((np.mean(xs), np.mean(
            ys), np.mean(zs), np.std(xs), np.std(ys), np.std(zs)))
        feature_vector = np.hstack((feature_vector, np.mean(abs(
            xs - np.mean(xs))), np.mean(abs(ys - np.mean(ys))), np.mean(abs(zs - np.mean(zs)))))
        feature_vector = np.hstack((feature_vector, np.mean(
            np.sqrt(np.power(xs, 2) + np.power(ys, 2) + np.power(zs, 2)))))
        feature_vector = np.hstack((feature_vector, np.divide(np.histogram(xs, 10), n)[
                                   0], np.divide(np.histogram(ys, 10), n)[0], np.divide(np.histogram(zs, 10), n)[0]))
        return feature_vector

    # filename='shelve.out'
    def ExtractAndSaveDSasExperiments(self,filename):
        my_shelve = shelve.open(filename, 'n')
        # readlabels
        exp_id, usr_id, act_id, act_be, act_en = self.ReadLabels()
        # readAllExperiments
        self.ReadAllExperiments(exp_id, usr_id, act_id, act_be, act_en)
        my_shelve['_experiments'] = self._experiments

    def SplitAllExpermintsAsSegmentSize(self,segment_size, _trainingExperiments, _testExperiments,filename):
        # SplitTrainigTest
        # SplitTrainTest(test_user_ids)
        # SplitExpintoSegment
        # training data
            x = []          
            gyro_x = []     
            y = []          
            gyro_y = []     
            z = []          
            gyro_z = []     
            features =[]     
            answers_raw = []

            # testing data

            test_x = []         
            test_gyro_x = []    
            test_y = []         
            test_gyro_y = []    
            test_z = []         
            test_gyro_z = []    
            test_features =   []
            test_answers_raw = []
            i=0
            for expr in _trainingExperiments:
                trainingexprdata = self.SplitExprIntoSegmentSize(expr, segment_size)
                x = np.hstack((x, trainingexprdata.paramX))
                y = np.hstack((y, trainingexprdata.paramY))
                z = np.hstack((z, trainingexprdata.paramZ))
                features+=trainingexprdata.paramFeatures

                gyro_x = np.hstack((gyro_x, trainingexprdata.paramGX))
                gyro_y = np.hstack((gyro_y, trainingexprdata.paramGY))
                gyro_z = np.hstack((gyro_z, trainingexprdata.paramGZ))

                answers_raw = np.hstack((answers_raw, trainingexprdata.ParamAnswers_raw))

            trainingData = StructData(
                x, y, z, gyro_x, gyro_y, gyro_z, features, answers_raw)
            for expr in _testExperiments:
                testingexprdata = self.SplitExprIntoSegmentSize(expr,segment_size)
                test_x = np.hstack((test_x, testingexprdata.paramX))
                test_y = np.hstack((test_y, testingexprdata.paramY))
                test_z = np.hstack((test_z, testingexprdata.paramZ))
                test_features+=testingexprdata.paramFeatures


                test_gyro_x = np.hstack((test_gyro_x, testingexprdata.paramGX))
                test_gyro_y = np.hstack((test_gyro_y, testingexprdata.paramGY))
                test_gyro_z = np.hstack((test_gyro_z, testingexprdata.paramGZ))

                test_answers_raw = np.hstack((test_answers_raw, testingexprdata.ParamAnswers_raw))

            testingData = StructData(test_x, test_y, test_z, test_gyro_x,
                                    test_gyro_y, test_gyro_z, test_features, test_answers_raw)
            answer_vector = []
            for i in range(len(trainingData.ParamAnswers_raw)):
                vect = np.zeros((1, 6))[0]
                vect[np.int(trainingData.ParamAnswers_raw[i])-1] = 1
                answer_vector = np.hstack((answer_vector, vect))
            test_answ_vector = []
            for i in range(len(testingData.ParamAnswers_raw)):
                vect = np.zeros((1, 6))[0]
                vect[np.int((testingData.ParamAnswers_raw[i]))-1] = 1
                test_answ_vector = np.hstack((test_answ_vector, vect))
            all_data = np.vstack((trainingData.paramX, trainingData.paramY, trainingData.paramZ,
                                trainingData.paramGX, trainingData.paramGY, trainingData.paramGZ))
            all_data = all_data.T[np.newaxis][0]
            all_test_data = np.vstack((testingData.paramX, testingData.paramY, testingData.paramZ,
                                    testingData.paramGX, testingData.paramGY, testingData.paramGZ))
            all_test_data = all_test_data.T[np.newaxis][0]

            #Normlize Features
            trainingData.paramFeatures = trainingData.paramFeatures - np.mean(trainingData.paramFeatures, axis = 0)
            trainingData.paramFeatures = trainingData.paramFeatures / np.std(trainingData.paramFeatures, axis = 0)

            testingData.paramFeatures = testingData.paramFeatures - np.mean(testingData.paramFeatures, axis = 0)
            testingData.paramFeatures = testingData.paramFeatures / np.std(testingData.paramFeatures, axis = 0)

            my_shelve=shelve.open(filename,'n')
            my_shelve['data_train']=all_data
            my_shelve['features']=trainingData.paramFeatures
            my_shelve['data_test']=all_test_data
            my_shelve['features_test']=testingData.paramFeatures
            my_shelve['labels_train']=answer_vector
            my_shelve['labels_test']=test_answ_vector

    def PreparingData(self,segment_size=128):
        shelvePreparedData="ShelvePreparedDataSegSiz/"+str(segment_size)
        if not os.path.exists(shelvePreparedData+".bak"):
            _trainingExperiments, _testExperiments = self.SplitTrainTest(self._test_user_ids,self._experiments)
            self.SplitAllExpermintsAsSegmentSize(segment_size,_trainingExperiments, _testExperiments,shelvePreparedData)
        return shelvePreparedData