from __future__ import division
import os
import email
import bs4
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import log


# Data Preparation

def get_messages(folder):
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), errors='ignore') as eml:
            yield email.message_from_file(eml)

def extract_body(message):
    if message.is_multipart():
        return '\n'.join(map(extract_body, message.get_payload()))

    return flatten_message_to_text(message)

def flatten_message_to_text(message):
    if message.get_content_type() == 'text/html':
        return bs4.BeautifulSoup(message.get_payload(), 'html.parser').get_text()

    return message.get_payload()

def normalize_text(text):
    words = nltk.word_tokenize(text.lower())
    filtered = filter(lambda word: word not in nltk.corpus.stopwords.words('english'), words)

    return ' '.join(filtered) + ' '

def get_data_frame_from_train_messages():
    notspam_messages = [normalize_text(extract_body(m)) for m in get_messages('./data/notSpam')]
    spam_messages = [normalize_text(extract_body(m)) for m in get_messages('./data/spam')]

    return pd.DataFrame({
        'message': notspam_messages + spam_messages,
        'spam': [0 for _ in range(len(notspam_messages))] +
            [1 for _ in range(len(spam_messages))]
    })

def get_data_frame_from_unknown_messages():
    filenames = os.listdir('./data/unknown')
    unknown_messages = [normalize_text(extract_body(m)) for m in get_messages('./data/unknown')]

    return pd.DataFrame({
        'filename': filenames,
        'message': unknown_messages
    })

# Naive Bayes implementation

def fit(train_set):
    classes, frequencies = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for features, label in train_set:
        classes[label] += 1
        for feature in features:
            frequencies[label, feature] += 1

    for label, feature in frequencies:
        frequencies[label, feature] /= classes[label]

    for cl in classes:
        classes[cl] /= len(train_set)

    return classes, frequencies

def classify(classifier, features):
    classes, probabilities = classifier
    log_metrics = lambda cl: -log(classes[cl]) \
        + sum(-log(probabilities.get((cl, ft), 10 ** (-7))) for ft in features)

    return min(classes.keys(), key = log_metrics)

def predict(classifier, predict_set):
    return [classify(classifier, message.split()) for message in predict_set]


if __name__ == '__main__':
    # get_data_frame_from_train_messages().to_csv('./data/train.csv', index_label=False)
    # get_data_frame_from_unknown_messages().to_csv('./data/unknown.csv', index_label=False)

    train_df = pd.read_csv('./data/train.csv')
    # unknown_df = pd.read_csv('./data/unknown.csv')

    message_train, message_test, label_train, label_test = train_test_split(train_df.message, train_df.spam,
                                                                            test_size=0.3, shuffle=True)
    message_train_features = \
        [(message.split(), label) for message, label in zip(message_train, label_train)]

    classifier = fit(message_train_features)

    predicted = predict(classifier, message_test)

    accuracy = np.mean(predicted == label_test)
    print(accuracy)

    # predicted = predict(classifier, unknown_df.message)
    # pd.DataFrame({
    #     'filename': unknown_df.filename,
    #     'spam': predicted
    # }).to_csv('unknown_predicted.csv', index=False)

