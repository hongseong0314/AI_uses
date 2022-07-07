import numpy as np 
import pandas as pd 
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

class Rating_model():
    MODEL_NAME = {"xgb":XGBClassifier, "rfc":RandomForestClassifier, "gb":GaussianNB, "knn":KNeighborsClassifier} 
    def __init__(self):

        self.model_list = {}

    def normalize(self, x_train, y_train):
        x_scaler = MinMaxScaler()
        x_scaler.fit(x_train)

        y_scaler = MinMaxScaler()
        y_scaler.fit(y_train)

        self.x_s = x_scaler
        self.y_s = y_scaler
        pass

    def model_append(self, model_name, model_clf):
        self.model_list[model_name] = self.MODEL_NAME[model_name](**model_clf)

    def fit(self, model_name, x_train, y_train, x_test, y_test, cv, search_cfg):
        if self.x_s == None:
            self.normalize(x_train, y_train)
        x_train = self.x_s.transform(x_train)
        y_train = self.y_s.transform(x_train)

        clf = self.model_list[model_name]

        random_clf = RandomizedSearchCV(clf, 
                                        param_distributions=search_cfg, 
                                        n_jobs=-1, n_iter=10,
                                        scoring = "accuracy" ,
                                        cv=cv, 
                                        verbose=2 
                                        )
        random_clf.fit(x_train, y_train)
        print(f"best acc : {random_clf.best_score_}")
        
        skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        #cross_val_score(clf, X_train, y_train_, cv=skfolds, scoring="accuracy") #sv에 계증적 샘플링 넣어준다.