import numpy as np 
import pandas as pd 
import os

from codes.utills import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform

from codes.model import Rating_model

import warnings
warnings.filterwarnings('ignore')

def main():
    # csv 경로
    path = os.path.join(os.getcwd(), 'german_credit_data.csv')

    # 데이터 프레임으로 불러오기
    df = pd.read_csv(path, index_col='Unnamed: 0')

    # 데이터 전처리
    df_clean = preprocessing(df)

    # target 변수 Risk 제거
    x = df_clean.drop(columns = ['Risk']).to_numpy()
    # 타겟 변수만 저장
    y = df_clean['Risk']
    y = y.to_numpy().ravel() # 1 차원 벡터 형태로 출력하기 위해 ravel 사용

    # 홀드아웃 검증 0.8 : 0.2 비율
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify=y)

    # Rating 모델 선언
    model = Rating_model()
    model_list = ["xgb", "rfc", "svc", "knn"]

    # 파라미터 설정
    xgb_config = {
        'height':1, 
        'grid' :True, 
        'importance_type':'gain', 
        'show_values':False, 
        'max_num_features':20
    }

    rfc_config = {
            'n_estimators':100, 
            'max_depth':12,
            'min_samples_leaf':8,
            'min_samples_split':8,
            'random_state':0,
    }

    svc_conig = {
        'kernel':"rbf",
        'probability':True,
    }

    knn_conig = {
        'n_neighbors' : 5,
    }
        
    # 하이퍼 파라미터 서치
    xgb_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=26),
        }

    rfc_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=26),
            'max_depth':randint(low=1, high=25),
        }

    svc_distribs = {
        'kernel':["rbf", 'poly'],
        'gamma':uniform(0, 5),
        'C':uniform(0, 1),
    }

    knn_distribs = {
            'n_neighbors': randint(low=3, high=7),
        }


    model_config = [xgb_config, rfc_config, svc_conig, knn_conig]
    model_distribs = [xgb_distribs, rfc_distribs, svc_distribs, knn_distribs]

    # 모델 선언
    for model_name, model_cfg in zip(model_list, model_config):
        model.model_append(model_name, model_cfg)

    # 모델 학습
    for model_name, model_distribs in zip(model_list, model_distribs):
        model.fit(model_name, x_train, y_train, 3, 10, model_distribs)

if __name__ == '__main__':
    print("train start...")
    main()