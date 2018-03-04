import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import SVC


data = pd.read_csv('./data/train.csv', delimiter=';')

data_train, data_predict = data[data.pidAggr.notna()], data[data.pidAggr.isna()]
data_train.drop(axis=1, labels=['id','dem_cand', 'rep_cand'], inplace=True)

data_train, data_test, answer_train, answer_test = train_test_split(data_train.drop(axis=1, labels='pidAggr'),
                                                                    data_train.pidAggr,
                                                                    test_size=0.7, shuffle=True)
# data_train, data_validation, answer_train, answer_validation = train_test_split(data_train,
#                                                                                 answer_train,
#                                                                                 test_size=0.2, shuffle=True)

model = SGDClassifier()

model.fit(data_train, answer_train)

predicted = model.predict(data_test)
print(np.mean(predicted == answer_test))

print(cross_val_score(model, data_train, answer_train, cv=6))