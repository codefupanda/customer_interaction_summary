# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import json

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
from tensorflow.keras.layers import Embedding, Flatten, Dense, Softmax
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import models
from model_configs import model_configs
# from models import Net

import kerastuner
from kerastuner.tuners import RandomSearch


@click.command()
@click.option('--input_filepath', default='data/processed', type=click.Path())
@click.option('--output_filepath', default='models/', type=click.Path())
@click.option('--pad_sequences_maxlen', default=1000, type=int)
@click.option('--max_words', default=10000, type=int)
@click.option('--epochs', default=10, type=int)
@click.option('--batch_size', default=32, type=int)
@click.option('--output_dim', default=32, type=int)
def main(input_filepath, output_filepath, pad_sequences_maxlen, max_words, epochs, batch_size, output_dim):
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

    isear = pd.read_csv(input_filepath + '/isear_train.csv',
                        sep='|', error_bad_lines=False, usecols=['SIT', 'EMOT'])
    isear['SIT'] = isear['SIT'].astype(str)
    isear_test = pd.read_csv(input_filepath + '/isear_test.csv',
                             sep='|', error_bad_lines=False, usecols=['SIT', 'EMOT'])
    sit_train, sit_test, emot_train, emot_test = train_test_split(
        isear['SIT'], isear['EMOT'], test_size=0.1)
    x_test_original, y_test = isear_test['SIT'], isear_test['EMOT']
    number_of_classes = len(isear.EMOT.unique())

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(sit_train)
    sequences_train = tokenizer.texts_to_sequences(sit_train)
    sequences_test = tokenizer.texts_to_sequences(sit_test)
    x_test = tokenizer.texts_to_sequences(x_test_original)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    sequences_train = pad_sequences(
        sequences_train, maxlen=pad_sequences_maxlen, padding='post')
    sequences_test = pad_sequences(
        sequences_test, maxlen=pad_sequences_maxlen, padding='post')
    x_test = pad_sequences(x_test, maxlen=pad_sequences_maxlen, padding='post')
    embedding_matrix_glove = get_embedding_matrix(
        max_words, word_index, output_dim)

    ##### BERT #####
    FullTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    bert_tokenizer = FullTokenizer(vocab_file, do_lower_case)
    bert_sequences_train = bert_encode(sit_train, bert_tokenizer, pad_sequences_maxlen)
    bert_sequences_test = bert_encode(sit_test, bert_tokenizer, pad_sequences_maxlen)
    bert_x_test = bert_encode(x_test_original, bert_tokenizer, pad_sequences_maxlen)
    ##### BERT ENDS #####

    reports = {}
    for model_config in model_configs:
        print('Training the model: ' + model_config)
        class_name = model_configs[model_config]['class_name']
        params = model_configs[model_config]['params']
        params['pad_sequences_maxlen'] = pad_sequences_maxlen
        params['max_words'] = max_words
        params['number_of_classes'] = number_of_classes
        params['embedding_matrix'] = embedding_matrix_glove if model_configs[model_config]['meta']['include_glove'] else None
        params['embedding_matrix'] = 'bert' if model_configs[model_config]['meta']['include_bert'] else None

        if params['embedding_matrix'] == 'bert':
            print("ghjgj - " + str(len(bert_sequences_train[0])) + str(len(bert_sequences_train[1])))
            model, hyperparameters = train_models(
                bert_sequences_train, bert_sequences_test, emot_train, emot_test, class_name, epochs, batch_size, params)
            y_pred = model.predict(bert_x_test)
        else:
            model, hyperparameters = train_models(
                sequences_train, sequences_test, emot_train, emot_test, class_name, epochs, batch_size, params)
            y_pred = model.predict(x_test)
        y_pred = y_pred.argmax(axis=-1)
        model_scores = pd.DataFrame(
            classification_report(y_pred, y_test, output_dict=True))
        model.save(output_filepath + model_config)
        with open(output_filepath + model_config + "_hyperparameters.json", 'w') as file:
            # use `json.loads` to do the reverse
            file.write(json.dumps(hyperparameters.values))
        reports[model_config] = model_scores
    final_report = pd.concat(reports.values(), keys=reports.keys())
    final_report.to_csv(output_filepath + "/final_report.csv")
    print(final_report)


def train_models(x_train, x_test, y_train, y_test, model_name, epochs, batch_size, params):
    # Get the class object from the models file and create instance
    model = getattr(models, model_name)(**params)
    tuner = RandomSearch(
        model,
        objective=kerastuner.Objective("val_f1_m", direction="max"),
        max_trials=5,
        executions_per_trial=1,
        directory='random_search',
        project_name='sentiment_analysis_' + str(model_name),
        distribution_strategy=tf.distribute.MirroredStrategy()
    )
    tuner.search_space_summary()
    tuner.search(x_train, to_categorical(y_train), epochs=epochs,
                 validation_data=(x_test, to_categorical(y_test)))
    return tuner.get_best_models(num_models=1)[0], tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters


def get_embedding_matrix(max_words, word_index, output_dim=50):
    glove_dir = 'data/external'
    embeddings_index = {}

    f = open(os.path.join(glove_dir, 'glove.6B.' + str(output_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((max_words, output_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def bert_encode(texts, tokenizer, max_len=1000):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
