import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def gain_train_data():
    df = pd.read_csv('./data/train.csv', delimiter='\t').dropna()

    return train_test_split(df.name, df.isOrg, test_size=0.3)

def create_model():
    return Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-5))
    ])

def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    people_df = test_df[~test_df.isOrg]
    organisations_df = test_df[test_df.isOrg]

    predicted_people = model.predict(people_df.name)
    predicted_organisations = model.predict(organisations_df.name)

    people_accuracy = np.mean(predicted_people == people_df.isOrg)
    organisations_accuracy = np.mean(predicted_organisations == organisations_df.isOrg)

    accuracy = math.sqrt(people_accuracy * organisations_accuracy)
    print(accuracy, people_accuracy, organisations_accuracy)


def predict_and_write(model, data_predict, filename):
    preprocessed = data_predict.name.apply(lambda name: '' if pd.isnull(name) else name)

    predicted = pd.Series(data=model.predict(preprocessed), name='prediction')

    predicted_df = pd.concat([data_predict.name, predicted], axis=1)
    predicted_df.to_csv(filename, sep='\t')


if __name__ == '__main__':
    data_train, data_test, answer_train, answer_test = gain_train_data()

    model = create_model()
    model.fit(data_train, answer_train)

    validate(model, data_test, answer_test)

    data_predict = pd.read_csv('./data/test.csv', delimiter='\t')
    predict_and_write(model, data_predict, './data/predicted.csv')
