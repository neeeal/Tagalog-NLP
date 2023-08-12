#### TWEETS-SENTIMENT-ANAYLSIS

import csv
import numpy as np
from os.path import join
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# INITALIZE GLOBAL VARIABLES
oov_token = '<OOV>'
batch_size = 32
train_size = 10000
test_size = 2000

# DATA PREPROCESSING

## Get raw data from csv files
data_filepath = 'trainingandtestdata'
train_filepath = join(data_filepath,'training.1600000.processed.noemoticon.csv')
test_filepath = join(data_filepath,'testdata.manual.2009.06.14.csv')

def read_csv(filepath):
    data = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    return data

train_data = read_csv(train_filepath)
test_data = read_csv(test_filepath)

## separate values and labels
def split_data(data):
    values = []
    labels = []
    for row in data:
        values.append(row[-1])
##        labels.append(row[0]) ## not needed
    return values, labels

train_corpus, _ = split_data(train_data[:train_size])
test_corpus, _ = split_data(test_data[:test_size])

## Initialize tokenizer
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(train_corpus)
total_words = len(tokenizer.word_index) + 1

## generate n_grams_seqs
def generate_n_grams_seqs(corpus, n_grams):
    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seqs = token_list[:i+1]
            input_sequences.append(n_gram_seqs)

    return input_sequences

train_input_sequences = generate_n_grams_seqs(train_corpus, tokenizer)
test_input_sequences = generate_n_grams_seqs(test_corpus, tokenizer)

train_max_sequence_length = max([len(seq) for seq in train_input_sequences])
test_max_sequence_length = max([len(seq) for seq in test_input_sequences])

## pad generated sequences
def pad_seqs(sequences, maxlen):
    padded_sequences = pad_sequences(sequences=sequences, maxlen=maxlen)

    return padded_sequences

train_input_sequences = pad_seqs(train_input_sequences, maxlen=train_max_sequence_length)
test_input_sequences = pad_seqs(test_input_sequences, maxlen=test_max_sequence_length)

## split to features and labels
def features_and_labels(input_sequences, total_words):
    features = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    categorical_labels = to_categorical(labels, num_classes=total_words)

    return features, categorical_labels

train_features, train_labels = features_and_labels(train_input_sequences, total_words)
test_features, test_labels = features_and_labels(test_input_sequences, total_words)

## create model
