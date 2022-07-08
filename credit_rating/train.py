import numpy as np 
import pandas as pd 
import os

from code.utills import preprocessing
from sklearn.model_selection import train_test_split

from code.model import Rating_model
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

# 홀드아웃 검증 0.8 : 0.2 비율
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify=y)

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

knn_conig = {
    'n_neighbors' : 5,
}

gb_conig = {

}

model_list = ["xgb", "rfc", "gb", "knn"]
model_config = [xgb_config, rfc_config, knn_conig, gb_conig]
model = Rating_model()

for model_name, model_cfg in zip(model_list, model_config):
    model.model_append(model_name, model_cfg)