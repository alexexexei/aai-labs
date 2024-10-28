import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def load_dataset(path: str, label_dict):
    train_X = read_data(path, 'train.csv').values[:,:-2]
    train_y = read_data(path, 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data(path, 'test.csv').values[:,:-2]
    test_y = read_data(path, 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return (train_X, train_y, test_X, test_y)


path = 'aai_lab1/'

df = read_data(path, 'train.csv')
df.head()

label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}
train_X, train_y, test_X, test_y = load_dataset(path, label_dict)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_X, train_y)

yhat = model.predict(test_X)

target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

print(classification_report(test_y, yhat, target_names=target_names))