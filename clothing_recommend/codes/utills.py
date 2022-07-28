import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
import contractions
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess(remove_stopwords=True):
    
    # 문장 전처리
    def preprocess_sentence(sentence):
        sentence = sentence.lower() 
        sentence = re.sub(r'\([^)]*\)', '', sentence) 
        sentence = re.sub('"','', sentence) 
        sentence = ' '.join([contractions.fix(t) for t in sentence.split(" ")]) 
        sentence = re.sub(r"'s\b","",sentence) 
        sentence = re.sub("[^a-zA-Z]", " ", sentence) 
        sentence = re.sub('[m]{2,}', 'mm', sentence)

        # 불용어 제거 (Text)
        if remove_stopwords:
            tokens = ' '.join(word for word in sentence.split() if not word in en_stops if len(word) > 1)
        # 불용어 미제거 (Summary)
        else:
            tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
        return tokens

    # 데이터 불러오기
    path = r'data/Womens Clothing E-Commerce Reviews.csv'
    data = pd.read_csv(path, index_col="Unnamed: 0")
    
    # review 데이터 중복값 및 결측값 제거
    data.drop_duplicates(subset=['Review Text'], inplace=True)
    df_clean = data[~data['Review Text'].isnull()]
    
    # 단어 정규식 및 stop word 정제
    en_stops = set(stopwords.words('english'))
    clean_review = df_clean['Review Text'].apply(lambda x : preprocess_sentence(x))
    
    del data, en_stops
    return clean_review, df_clean['Recommended IND'].to_numpy()

def word_tokenizer(sentence, src_vocab=5000, max_len=50):
    src_tokenizer = Tokenizer(num_words = src_vocab) 
    src_tokenizer.fit_on_texts(sentence)

    # token 화
    token_X = src_tokenizer.texts_to_sequences(sentence) 
    
    # padding
    token_X = pad_sequences(token_X, maxlen = max_len, padding='post')
    return token_X