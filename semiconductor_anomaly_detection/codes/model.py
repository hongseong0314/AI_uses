import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

import joblib
from scipy.stats import randint
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, f1_score
from sklearn.metrics import roc_curve

from codes.utills import downsampling

class Anomaly_Detection_model():
    MODEL_NAME = {"xgb":XGBClassifier, "lda":LinearDiscriminantAnalysis, "svc":SVC, "knn":KNeighborsClassifier} 
    def __init__(self):

        self.model_list = {}
        self.results = {}
        self.x_s = None
        self.y_s = None

    def normalize(self, x_train, y_train):
        x_scaler = MinMaxScaler()
        x_scaler.fit(x_train)

        y_scaler = MinMaxScaler()
        y_scaler.fit(y_train)

        self.x_s = x_scaler
        self.y_s = y_scaler

        create_directory("scale")
        joblib.dump(x_scaler, "scale/x.save")
        joblib.dump(y_scaler, "scale/y.save")
        pass
    
    def load_scale(self):
        self.x_s = joblib.load("scale/x.save")
        self.y_s = joblib.load("scale/y.save")

    def model_append(self, model_name, model_clf=None):
        model = self.MODEL_NAME[model_name]
        self.model_list[model_name] = model(**model_clf)

    def fit(self, model_name, x_train, y_train, bagging_num, model_clf):
        if y_train.ndim == 1:
            y_train = y_train.values.reshape(-1,1)
            
        if self.x_s is None:
            self.normalize(x_train, y_train)
            
        x_train = self.x_s.transform(x_train)
        y_train = self.y_s.transform(y_train)

        # random seed
        random_seed = np.random.randint(0, 100, bagging_num)
        
        # model save path
        create_directory("weight")
        create_directory("weight/"+model_name)

        model_list = []
        for b, seed in enumerate(random_seed):
            x, y = downsampling(x_train, y_train, seed)
            model = self.MODEL_NAME[model_name]
            clf = model(**model_clf)
            clf.fit(x,y)
            model_list.append(clf)
        
            print("-"*50)
            print("model name : {} bagging num : {} acc : {}".format(model_name, b, 100))

            #save  
            save_path = 'weight/' + model_name + '/model_{}.pkl'.format(b)
            joblib.dump(clf, save_path) 

        self.model_list[model_name] = model_list
    
    def predict(self, model_name, x_test, mode='prob'):
        model_list = self.model_list[model_name]
        alpha = 1/len(model_list)

        if self.x_s == None:
            self.load_scale()

        x_test = self.x_s.transform(x_test)
        for i, model in enumerate(model_list):
            if i == 0:
                pred = alpha * model.predict_proba(x_test)
            else:
                pred += alpha * model.predict_proba(x_test)

        if mode=='prob':
            return pred
        else:
            return np.argmax(pred, axis=1)
    
    def evals(self, model_name ,x_test, y_test, thresholds=[0.5]):
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        y_test = np.clip(y_test, 0, 1)
        pred = self.predict(model_name, x_test, 'prob')
        pred_1 = pred[:,1].reshape(-1,1)
        self.get_eval_by_threshold(y_test, pred_1, thresholds)

    
    def get_clf_eval(self, y_test , pred):
        confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test , pred)
        precision = precision_score(y_test , pred)
        recall = recall_score(y_test , pred)
        f1 = f1_score(y_test,pred)
        print('오차 행렬')
        print(confusion)
        print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))

    
    def get_eval_by_threshold(self, y_test , pred_proba_c1, thresholds):
        # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
        for custom_threshold in thresholds:
            binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
            custom_predict = binarizer.transform(pred_proba_c1)
            print('임곗값:',custom_threshold)
            self.get_clf_eval(y_test , custom_predict)
            print('-'*50)

    def roc_curve_plot(self, model_list, x_test, y_test):
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        y_test = np.clip(y_test, 0, 1)
        
        # 모델별 임계값에 따른 FPR, TPR 값을 변환
        with plt.style.context('ggplot'):
            plt.figure(figsize=(8,6))
            for model_name in model_list:
                pred_proba_c1 = self.predict(model_name, x_test.copy(), 'prob')
                pred_proba_c1 = pred_proba_c1[:,1].reshape(-1,1)
                fprs, tprs, threshodls = roc_curve(y_test, pred_proba_c1)

                # ROC Curve를 plot 곡선으로 시각화
                plt.plot(fprs, tprs, label=model_name)
            plt.plot([0,1],[0,1], 'k--', label = 'Random')
            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1),2))
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('FPR(1-Sensitivity)')
            plt.ylabel('TPR(Recall)')
            plt.legend()
            plt.show()
    
    def load_model(self, model_name):
        path = glob('weight/' + model_name + '/*')
        model_list = [joblib.load(p) for p in path]
        self.model_list[model_name] = model_list

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)