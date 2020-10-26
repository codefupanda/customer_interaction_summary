# The basics
import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Softmax, LSTM, SimpleRNN, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, Dropout

# from keras_self_attention import SeqSelfAttention

# Hyperparameter tunning 
from kerastuner import HyperModel


class DNNModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(DNNModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Flatten())
        model.add(Dropout(dropout))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class SimpleRNNModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(SimpleRNNModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(SimpleRNN(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class CNNModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(CNNModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

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
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class LSTMModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(LSTMModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class StackedLSTMModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(StackedLSTMModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(LSTM(64, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class BiLSTMModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(BiLSTMModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class StackedBiLSTMModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(StackedBiLSTMModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Flatten())
        model.add(Dense(number_of_classes + 1,  activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


class HybridModel(HyperModel):

    def __init__(self, pad_sequences_maxlen, max_words, number_of_classes, epochs=None, output_dim=50, embedding_matrix=None, dropout=None, recurrent_dropout=None):
        super(HybridModel, self).__init__()
        self.dropout=dropout
        self.epochs = epochs
        self.max_words=max_words
        self.output_dim=output_dim
        self.embedding_matrix=embedding_matrix
        self.recurrent_dropout=recurrent_dropout
        self.number_of_classes=number_of_classes
        self.pad_sequences_maxlen=pad_sequences_maxlen
        self.initial_learning_rate = None #initial_learning_rate

    def build(self, hp):
        pad_sequences_maxlen = self.pad_sequences_maxlen
        max_words = self.max_words
        number_of_classes = self.number_of_classes
        epochs = self.epochs if self.epochs else hp.Int('epochs', min_value=10, max_value=20, default=15, step=5)
        output_dim = self.output_dim
        embedding_matrix = self.embedding_matrix
        initial_learning_rate = self.initial_learning_rate if self.initial_learning_rate else hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, default=0.01, step=0.003)
        dropout =  self.dropout if self.dropout else hp.Float('dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)
        recurrent_dropout = self.recurrent_dropout if self.recurrent_dropout else hp.Float('recurrent_dropout', min_value=0.0, max_value=0.4, default=0.3, step=0.1)

        model = Sequential()
        if embedding_matrix is not None:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen, weights=[embedding_matrix], trainable=False))
        else:
            model.add(Embedding(max_words, output_dim=output_dim, input_length=pad_sequences_maxlen))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Dropout(dropout))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(number_of_classes + 1, activation='softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=20,
            decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule, decay=initial_learning_rate/epochs)
        # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_m])
        return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
