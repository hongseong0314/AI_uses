import numpy as np 
import pandas as pd 

def preprocessing(df):
    df_clean = df.copy()
    df_clean['Saving accounts'] = df_clean['Saving accounts'].fillna('Others')
    df_clean['Checking account'] = df_clean['Checking account'].fillna('Others')
    
    # 2개의 카테고리를 갖는 데이터는 replace를 사용하여 0,1로 변환 합니다.
    df_clean = df_clean.replace(['good','bad'],[0,1])

    # object 자료형 데이터의 변수를 정리합니다.
    cat_features = ['Sex','Housing', 'Saving accounts', 'Checking account','Purpose']
    # 수치 자료형 데이터의 변수를 정리합니다.
    num_features=['Age', 'Job', 'Credit amount', 'Duration','Risk']

    # 더미를 기법을 사용하여 변환합니다.
    for variable in cat_features:
        # pandas의 더미 방식을 사용하여 object 자료형 데이터를 변환한 dataframe을 생성합니다.
        dummies = pd.get_dummies(df_clean[cat_features])
        # 기존 수치형 데이터에 더미로 새로 생성된 데이터를 추가합니다.
        df1= pd.concat([df_clean[num_features], dummies],axis=1)

    return df1