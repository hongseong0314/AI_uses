import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt

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

class Rating_model():
    MODEL_NAME = {"xgb":XGBClassifier, "rfc":RandomForestClassifier, "svc":SVC, "knn":KNeighborsClassifier} 
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
        pass

    def model_append(self, model_name, model_clf=None):
        model = self.MODEL_NAME[model_name]
        self.model_list[model_name] = model(**model_clf)

    def fit(self, model_name, x_train, y_train, cv, iters, search_cfg):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
            
        if self.x_s is None:
            self.normalize(x_train, y_train)
        x_train = self.x_s.transform(x_train)
        y_train = self.y_s.transform(y_train)

        clf = self.model_list[model_name]

        # search_clf
        random_clf = RandomizedSearchCV(clf, 
                                        param_distributions=search_cfg, 
                                        n_jobs=-1, n_iter=iters,
                                        scoring = "accuracy" ,
                                        cv=cv, 
                                        verbose=2 
                                        )
        random_clf.fit(x_train, y_train)
        print("-"*50)
        print(f"model name : {model_name}")
        print(f"best params : {random_clf.best_params_}")
        print(f"best acc : {random_clf.best_score_}")
        
        self.model_list[model_name] = random_clf.best_estimator_
        self.results[model_name] = random_clf.best_score_

        #save  
        create_directory("weight")
        create_directory("weight/"+model_name)
        save_path = 'weight/' + model_name + '/model.pkl'
        joblib.dump(random_clf.best_estimator_, save_path) 
    
    
    def evals(self, model_name ,x_test, y_test, thresholds=[0.5]):
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        x_test = self.x_s.transform(x_test)
        y_test = self.y_s.transform(y_test)
        model = self.model_list[model_name]
        pred = model.predict_proba(x_test)
        self.get_eval_by_threshold(y_test, pred[:,1].reshape(-1,1), thresholds)

    
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
        x_test = self.x_s.transform(x_test)
        y_test = self.y_s.transform(y_test)
        
        # 모델별 임계값에 따른 FPR, TPR 값을 변환
        with plt.style.context('ggplot'):
            plt.figure(figsize=(8,6))
            for model_name in model_list:
                model = self.model_list[model_name]
                pred_proba_c1 = model.predict_proba(x_test)[:, 1]
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
    
    def load_modal(self, model_name):
        path = 'weight/' + model_name + '/model.pkl'
        clf = joblib.load(path) 
        self.model_list[model_name] = clf

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)