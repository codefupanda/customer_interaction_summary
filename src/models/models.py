# The basics
import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Softmax, LSTM, SimpleRNN, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, Dropout

# from keras_self_attention import SeqSelfAttention

class DNNModel(Model):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, output_dim=50, embedding_matrix=None):
        super(DNNModel, self).__init__()
        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        self.model = model

    def call(self, inputs):
        return self.model(inputs)


class SimpleRNNModel(Model):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, output_dim=50, embedding_matrix=None):
        super(SimpleRNNModel, self).__init__()
        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        self.model = model

    def call(self, inputs):
        return self.model(inputs)


class CNNModel(Model):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, output_dim=50, embedding_matrix=None):
        super(CNNModel, self).__init__()
        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        self.model = model

    def call(self, inputs):
        return self.model(inputs)


class LSTMModel(Model):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, output_dim=50, embedding_matrix=None):
        super(LSTMModel, self).__init__()
        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(LSTM(128, return_sequences=True, dropout=0.2))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        self.model = model

    def call(self, inputs):
        return self.model(inputs)

class HybridModel(Model):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, output_dim=50, embedding_matrix=None):
        super(HybridModel, self).__init__()
        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Dropout(0.2))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(number_of_classes + 1, activation='softmax'))
        self.model = model

    def call(self, inputs):
        return self.model(inputs)
