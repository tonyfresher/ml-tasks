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


def gain_data():
    data = pd.read_csv('./data/train.csv', delimiter='\t').dropna()

    data.name = data[['name', 'isOrg']].apply(preprocess, axis=1)

    return train_test_split(data.name, data.isOrg, test_size=0.3)

def preprocess(row):
    name, is_organisation = row

    name = (name
        .replace('"', '')
        .lower())

    return name.split()[0] if is_organisation else name

def create_model():
    return Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer()),
        ('reg', RandomForestClassifier(n_estimators=15,
                                       criterion='entropy',
                                       min_samples_split=2,
                                       min_samples_leaf=2))
    ])


def validate(model, data_test, answer_test):    
    # "Accuracy" metrics checking
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
    predicted = model.predict(data_predict)

    with open(filename, 'w+') as output:
        predicted_table = zip(range(len(predicted)), data_predict, predicted)
        tsv_table = '\n'.join(f'{id}\t{name}\t{value}' for id, name, value in predicted_table)
        output.write(tsv_table)

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


if __name__ == '__main__':
    data_train, data_test, answer_train, answer_test = gain_data()

    model = create_model()
    features = \
        [(name.split(), label) for name, label in zip(data_train, answer_train)]

    model = fit(features)

    # parameters = {
    #     'reg__tol': [1e-5, 1e-3],
    #     'reg__penalty': ['l1', 'l2'],
    #     'reg__C': [1, 10]
    # }

    # gs_clf = GridSearchCV(model, parameters)
    # gs_clf = gs_clf.fit(data_train, answer_train)
    # print(gs_clf.best_score_, gs_clf.best_params_)

    validate(model, data_test, answer_test)
    # predict_and_write(model, data_predict, './data/predicted.csv')
    