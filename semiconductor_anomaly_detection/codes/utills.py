import numpy as np 
import pandas as pd 
import sys


# 수치형 범주형 나눠주기
def seperate_type(df):
    dtypes = df.dtypes
    numerical = [df.columns[idx] for idx, _ in enumerate(dtypes) if dtypes[idx] != object]
    category = [df.columns[idx] for idx, _ in enumerate(dtypes) if dtypes[idx] == object]
    
    return numerical, category

def preprocess(mode='train'):
    """
    데이터 전처리 함수
    """

    train = pd.read_csv('./data/uci-secom.csv')
    train = train.drop(columns = ['Time'], axis = 1)
    train_size = len(train)
    test = pd.read_csv('./data/uci-secom-test.csv')
    big_df = pd.concat([train, test], axis=0)


    # 데이터 결측값 처리
    big_df = big_df.replace(np.NaN, 0)
    #print(data.isnull().sum())
    
    # 데이터 타입별 나눠주기
    numerical_feature = seperate_type(big_df)[0]
    category_feature = seperate_type(big_df)[1]
    
    # 0만 있는 분포 제거
    corr_data = big_df[numerical_feature].corrwith(big_df['Pass/Fail']).sort_values(ascending=False)
    corr_df = pd.DataFrame(corr_data, columns=['Correlation'])
    NaN_col = corr_df.loc[corr_df['Correlation'].isnull()].index.tolist()
    big_df.drop(NaN_col, axis=1, inplace=True)
    if mode == 'train':
        data = big_df[:train_size]

    else:
        data = big_df[train_size:]

    # 타겟 데이터 분리
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]


    print("x data shape : {}".format(x.shape))
    print("y data shape : {}".format(y.shape))
    
    del corr_data, corr_df, big_df, train, test
    return x, y

def downsampling(x, y, seed=42):
    """
    데이터 다운 샘플링
    """
    fulls = np.hstack([x,y])
    data = pd.DataFrame(fulls)
    # Under sampling 수행(-1 label이 많기 때문에 클래스 균형을 맞춰준다)
    failed_tests = np.array(data[data.iloc[:, -1] == 1].index)
    no_failed_tests = len(failed_tests)

    normal_indices = data[data.iloc[:, -1] == 0]
    no_normal_indices = len(normal_indices)

    np.random.seed(seed)
    random_normal_indices = np.random.choice(no_normal_indices, size = no_failed_tests, replace = True)
    random_normal_indices = np.array(random_normal_indices)

    under_sample = np.concatenate([failed_tests, random_normal_indices])
    undersample_data = data.iloc[under_sample, :]

    x = undersample_data.iloc[:, :-1] 
    y = undersample_data.iloc[:, -1]
    y = np.ravel(y)

    # index shuffle
    permutation = list(np.random.permutation(x.shape[0]))
       
    del fulls, data, under_sample, normal_indices
    return x.iloc[permutation, :].to_numpy(), y[permutation].reshape(-1, 1)