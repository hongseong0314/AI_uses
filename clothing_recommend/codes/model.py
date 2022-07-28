import numpy as np
import pandas as pd
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from codes.utills import preprocess, word_tokenizer
from codes.transformer_block import TokenAndPositionEmbedding, TransformerBlock, MultiHeadAttention

class clothing_transformer():
    def __init__(self, max_len, vocab_size):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_model(self, embedding_dim, num_heads, dff, pretrained = False):
        if pretrained:
            self.model = tf.keras.models.load_model(pretrained)

        else:
            inputs = tf.keras.layers.Input(shape=(self.max_len,))
            embedding_layer = TokenAndPositionEmbedding(self.max_len, self.vocab_size, embedding_dim)
            x = embedding_layer(inputs)
            transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
            x = transformer_block(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(20, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
            
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        pass

    def trainer(self, X_train, y_train, X_test, y_test, embedding_dim, num_heads, dff, epoch, batch_size):
        self.get_model(embedding_dim, num_heads, dff)

        create_directory("weight")

        filename = 'weight/transfomer-epoch-{}-batch-{}.h5'.format(epoch, batch_size)
        checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                                    monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                                    verbose=1,            # 로그를 출력합니다
                                    save_best_only=True,  # 가장 best 값만 저장합니다
                                    mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                                    )

        earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss) 
                                    patience=3,         # 10회 Epoch동안 개선되지 않는다면 종료
                                    )

        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        history = self.model.fit(X_train, y_train, 
                                batch_size=batch_size, epochs=epoch, 
                                validation_data=(X_test, y_test), 
                                callbacks=[checkpoint, earlystopping])

        return history


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)