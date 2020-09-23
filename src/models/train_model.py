# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

# The basics
import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Flatten, Dense, Softmax

import models
from model_configs import model_configs
# from models import Net

@click.command()
@click.option('--input_filepath', default='data/raw', type=click.Path())
@click.option('--output_filepath', default='models/', type=click.Path())
@click.option('--pad_sequences_maxlen', default=1000, type=int)
@click.option('--max_words', default=10000, type=int)
@click.option('--epochs', default=20, type=int)
@click.option('--batch_size', default=32, type=int)
def main(input_filepath, output_filepath, pad_sequences_maxlen, max_words, epochs, batch_size):
    """ Train the models from the processed data (input_filepath) and persist the learned models in output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info('starting the training process')

    logger.info('--input_filepath ' + input_filepath)
    logger.info('--output_filepath ' + output_filepath)
    logger.info('--pad_sequences_maxlen ' + str(pad_sequences_maxlen))
    logger.info('--max_words ' + str(max_words))
    logger.info('--epochs ' + str(epochs))
    logger.info('--batch_size ' + str(batch_size))

    isear = pd.read_csv(input_filepath + '/isear.csv', sep='|', error_bad_lines=False, usecols=['Field1', 'SIT', 'EMOT'])
    number_of_classes = len(isear.EMOT.unique())

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(isear['SIT'])
    sequences = tokenizer.texts_to_sequences(isear['SIT'])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = pad_sequences(sequences, maxlen=pad_sequences_maxlen, padding='post')
    embedding_matrix_glove = get_embedding_matrix(max_words, word_index)
    
    reports = {}
    for model_config in model_configs:
        print('Training the model: ' + model_config)
        class_name = model_configs[model_config]['class_name']
        params = model_configs[model_config]['params']
        params['pad_sequences_maxlen'] = pad_sequences_maxlen
        params['max_words'] = max_words
        params['number_of_classes'] = number_of_classes
        params['embedding_matrix'] = embedding_matrix_glove
        model, model_scores = train_models(X, isear['EMOT'], class_name, epochs, batch_size, params)
        model.save(output_filepath + model_config)
        reports[model_config] = model_scores
    final_report = pd.concat(reports.values(), keys=reports.keys())
    final_report.to_csv(output_filepath + "/final_report.csv")
    print(final_report)

def train_models(X, y, model_name, epochs, batch_size, params):
    x_train, x_test, y_train, y_test = train_test_split(X, y)

        ## Get the class object from the models file and create instance
    model = getattr(models, model_name)(**params)

    opt = Adam(learning_rate=0.01)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, to_categorical(y_train),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, to_categorical(y_test)))
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=-1)

    return (model, pd.DataFrame(classification_report(y_pred, y_test, output_dict=True)))


def get_embedding_matrix(max_words, word_index):
    glove_dir = 'data/external'
    embeddings_index = {}

    f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_dim = 50 # if chaning this, update the file name above 

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
