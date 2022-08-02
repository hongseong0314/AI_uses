import numpy as np
import pandas as pd
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from codes.utills import preprocess, word_tokenizer
from codes.transformer_block import TokenAndPositionEmbedding, TransformerBlock

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
            self.model = tf.keras.models.load_model(pretrained, custom_objects={'TokenAndPositionEmbedding':TokenAndPositionEmbedding,
                                                                                'TransformerBlock':TransformerBlock,
                                                                                })

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
        checkpoint = ModelCheckpoint(filename,            
                                    monitor='val_loss',   
                                    verbose=1,           
                                    save_best_only=True,  
                                    mode='auto'          
                                    )

        earlystopping = EarlyStopping(monitor='val_loss',  
                                    patience=3,         
                                    )

        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        history = self.model.fit(X_train, y_train, 
                                batch_size=batch_size, epochs=epoch, 
                                validation_data=(X_test, y_test), 
                                callbacks=[checkpoint, earlystopping])

        return history
    
    def predict(self, x_test, prob=True):
        if prob:
            return self.model.predict(x_test)
        else:
            return np.argmax(self.model.predict(x_test), axis=1)

    def load_model(self, path):
        self.get_model(embedding_dim=None, num_heads=None, dff=None, pretrained=path)        


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)