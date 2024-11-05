import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def load_dataset(path: str, train_file: str, test_file: str, row: str,
                 label_dict):
    train_X = read_data(path, train_file).values[:, :-2]
    train_y = read_data(path, train_file)[row]
    train_y = train_y.map(label_dict).values
    test_X = read_data(path, test_file).values[:, :-2]
    test_y = read_data(path, test_file)
    test_y = test_y[row].map(label_dict).values
    return (train_X, train_y, test_X, test_y)


path = 'D:/code/aai-labs/aai_lab1/'
train_file = 'train.csv'
test_file = 'test.csv'
row = 'Activity'

df = read_data(path, train_file)
df.head()

label_dict = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1,
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}
train_X, train_y, test_X, test_y = load_dataset(path, train_file, test_file,
                                                row, label_dict)

model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(train_X, train_y)

yhat = model.predict(test_X)

target_names = [
    'Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing',
    'Laying'
]

print(classification_report(test_y, yhat, target_names=target_names))

conf_matrix = confusion_matrix(test_y, yhat)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=60)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
