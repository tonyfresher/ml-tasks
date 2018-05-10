import pandas as pd
import numpy as np
import math
from functools import reduce
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


def gain_train_data():
    data = pd.read_csv('./data/train.csv', delimiter='\t').dropna()

    data.name = data[['name', 'isOrg']].apply(lambda name: preprocess_train(name), axis=1)
    data = data.dropna()

    return train_test_split(data.name, data.isOrg, test_size=0.3)

def preprocess_train(row, analyzer=CountVectorizer().build_analyzer()):
    name, is_organisation = row

    if analyzer(name) == []:
        return np.nan

    name = name.replace('"', '').replace('-', ' ', 1).lower()

    return name.split()[0] if is_organisation else name

def preprocess_test(name):
    if pd.isnull(name):
        return ''
    
    return name.replace('"', '').replace('-', ' ', 1).lower()

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
    log_metrics = lambda cl: -math.log(classes[cl]) \
        + sum(-math.log(probabilities.get((cl, ft), 10 ** (-7))) for ft in features)

    return min(classes.keys(), key = log_metrics)

def predict(classifier, predict_set):
    return [classify(classifier, message.split()) for message in predict_set]

def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    people_df = test_df[~test_df.isOrg]
    organisations_df = test_df[test_df.isOrg]

    predicted_people = predict(model, people_df.name)
    predicted_organisations = predict(model, organisations_df.name)

    people_accuracy = np.mean(predicted_people == people_df.isOrg)
    organisations_accuracy = np.mean(predicted_organisations == organisations_df.isOrg)

    accuracy = math.sqrt(people_accuracy * organisations_accuracy)
    print(accuracy, people_accuracy, organisations_accuracy)


def predict_and_write(model, data_predict, filename):
    preprocessed = data_predict.name.apply(preprocess_test)

    predicted = pd.Series(data=predict(model, preprocessed), name='prediction')

    predicted_df = pd.concat([data_predict.name, predicted],
                             axis=1)

    predicted_df.to_csv(filename, sep='\t')


if __name__ == '__main__':
    data_train, data_test, answer_train, answer_test = gain_train_data()

    features = \
        [(name.split(), label) for name, label in zip(data_train, answer_train)]

    model = fit(features)

    validate(model, data_test, answer_test)

    # data_predict = pd.read_csv('./data/test.csv', delimiter='\t')
    # predict_and_write(model, data_predict, './data/predicted.csv')
