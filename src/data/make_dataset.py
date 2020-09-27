# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Sklearn
from sklearn.model_selection import train_test_split

# The basics
import numpy as np
import pandas as pd

import spacy

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
@click.argument('augment', type=bool, default=False)
@click.argument('remove-stopwords', type=bool, default=False)
def main(input_filepath, output_filepath, augment, remove_stopwords):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    sp = spacy.load('en_core_web_sm')

    isear = pd.read_csv(input_filepath + '/isear.csv', sep='|', error_bad_lines=False, usecols=['SIT', 'EMOT'])
    isear_train, isear_test = train_test_split(isear, stratify=isear['EMOT'])

    if remove_stopwords:
        nlp = spacy.load('en_core_web_sm')
        isear = isear.apply(lambda x: pd.Series([x[0], clean(nlp, x[1])]), axis=1)

    if augment:
        isear_train = augment(isear_train, isear)

    isear_train.to_csv(output_filepath + "/isear_train.csv", sep='|', index=False)
    isear_test.to_csv(output_filepath + "/isear_test.csv", sep='|', index=False)

    logger.info('successfully wrote train and test files')

def clean(nlp, strng):
    return " ".join([w.text for w in nlp(strng) if not (w.is_stop or w.is_punct)])

def augment(isear_train, isear):
    import nlpaug.augmenter.word as naw
    aug = naw.SpellingAug()
    isear_aug = isear_train.apply(lambda x: pd.Series([x[0], aug.augment(x[1])]), axis=1)
    isear_aug.columns = isear.columns
    isear_aug1 = isear_train.apply(lambda x: pd.Series([x[0], aug.augment(x[1])]), axis=1)
    isear_aug1.columns = isear.columns
    isear_train = pd.concat([isear_train, isear_aug, isear_aug1], ignore_index = True)
    return isear_train

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
