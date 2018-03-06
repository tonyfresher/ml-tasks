import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


# Reading and parsing data
data = pd.read_csv('./data/train.csv', delimiter=';')

data_train, data_predict = data[data.pidAggr.notna()], data[data.pidAggr.isna()]

data_train = data_train.query('pidAggr.notna() & pidAggr in [1, 2]')
data_train.drop(axis=1,
                labels=['id', 'dem_cand', 'rep_cand'] +
                [f'ISSUES_{n}' for n in range(1, 22)],
                inplace=True)

data_predict.drop(axis=1,
                  labels=['id', 'dem_cand', 'rep_cand', 'pidAggr'] +
                  [f'ISSUES_{n}' for n in range(1, 22)],
                  inplace=True)

# Splitting datasets
data_train, data_test, answer_train, answer_test = train_test_split(data_train.drop(axis=1, labels='pidAggr'),
                                                                    data_train.pidAggr,
                                                                    test_size=0.3, shuffle=True)
# Creating the model
model = Pipeline([
    ('scaler', RobustScaler()),
    ('forest', RandomForestClassifier(n_estimators=15,
                                      criterion='entropy',
                                      min_samples_split=2,
                                      min_samples_leaf=2))
])

# Learning
model.fit(data_train, answer_train)

# "Accuracy" metrics checking
predicted = model.predict(data_test)
print(np.mean(predicted == answer_test))

# Cross-validation checking
cross_val_score = cross_val_score(model, data_test, answer_test, cv=6)
print(cross_val_score.mean(), cross_val_score)

# Predicting
predicted = model.predict(data_predict)

with open('./data/predicted.csv', 'w+') as output:
    predicted_table = zip(range(len(predicted)), predicted)
    csv_table = '\n'.join(f'{id};{int(value)}' for id, value in predicted_table)
    output.write(csv_table)