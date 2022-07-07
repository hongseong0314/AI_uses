import numpy as np 
import pandas as pd 
import os

from code.utills import preprocessing
# from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

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

print(df_clean.info())

# # # 홀드아웃 검증 0.8 : 0.2 비율
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify=y)

# # # 모델 하이퍼 파리미터

# config = {
#      'height':1, 
#      'grid' :True, 
#      'importance_type':'gain', 
#      'show_values':False, 
#      'max_num_features':20
# }
# # 모델 정의
# model = XGBClassifier(**config)
# # 모델 학습
# model.fit(x_train, y_train)

# print("%s - train_score : %f, test score : %f" % (model.__class__.__name__
#                                             , model.score(x_train, y_train), 
#                                             model.score(x_test, y_test)))