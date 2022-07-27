import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re

def preprocess():
    # 데이터 불러오기
    path = r'data/Womens Clothing E-Commerce Reviews.csv'
    data = pd.read_csv(path, index_col="Unnamed: 0")
    
    # review 데이터 결측값 제거
    df_clean = data[~data['Review Text'].isnull()]
    
    # 단어 정규식 및 stop word 정제
    en_stops = set(stopwords.words('english'))
    clean_review = [[word for word in re.findall("[\w']+", row) if not word in en_stops]
                                for row in df_clean['Review Text'].str.lower()]
    del data, en_stops
    return clean_review, pd.Series(clean_review).apply(lambda x : " ".join(x)), df_clean['Recommended IND'].to_numpy()