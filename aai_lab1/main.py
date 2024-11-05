import os
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

# model 1
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(train_X, train_y)

yhat = model.predict(test_X)

target_names = [
    'Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing',
    'Laying'
]

print(classification_report(test_y, yhat, target_names=target_names))

conf_matrix = confusion_matrix(test_y, yhat)

# model 2
yhat_2 = KNeighborsClassifier(n_neighbors=3).fit(train_X, train_y).predict(test_X)

print(classification_report(test_y, yhat_2, target_names=target_names))

conf_matrix_2 = confusion_matrix(test_y, yhat_2)

# model 3
yhat_3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(train_X, train_y).predict(test_X)

print(classification_report(test_y, yhat_3, target_names=target_names))

conf_matrix_3 = confusion_matrix(test_y, yhat_3)

fig, axes = plt.subplots(1, 3, figsize=(20, 4))

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[0])
axes[0].set_title('LogisticRegression Confusion Matrix')
axes[0].set_xlabel('Predicted labels')
axes[0].set_ylabel('True labels')

sns.heatmap(conf_matrix_2, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[1])
axes[1].set_title('KNeighborsClassifier Confusion Matrix')
axes[1].set_xlabel('Predicted labels')
axes[1].set_ylabel('True labels')

sns.heatmap(conf_matrix_3, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[2])
axes[2].set_title('RandomForestClassifier Confusion Matrix')
axes[2].set_xlabel('Predicted labels')
axes[2].set_ylabel('True labels')

plt.tight_layout()
plt.show()

# model 2.2
yhat_2_2 = KNeighborsClassifier(n_neighbors=2).fit(train_X, train_y).predict(test_X)
conf_matrix_2_2 = confusion_matrix(test_y, yhat_2_2)
# model 2.3
yhat_2_3 = KNeighborsClassifier(n_neighbors=5).fit(train_X, train_y).predict(test_X)
conf_matrix_2_3 = confusion_matrix(test_y, yhat_2_3)

fig, axes = plt.subplots(1, 3, figsize=(20, 4))

sns.heatmap(conf_matrix_2_2, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[0])
axes[0].set_title('KNeighborsClassifier Confusion Matrix, n=2')
axes[0].set_xlabel('Predicted labels')
axes[0].set_ylabel('True labels')

sns.heatmap(conf_matrix_2, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[1])
axes[1].set_title('KNeighborsClassifier Confusion Matrix, n=3')
axes[1].set_xlabel('Predicted labels')
axes[1].set_ylabel('True labels')

sns.heatmap(conf_matrix_2_3, annot=True, fmt="d", cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=axes[2])
axes[2].set_title('KNeighborsClassifier Confusion Matrix, n=5')
axes[2].set_xlabel('Predicted labels')
axes[2].set_ylabel('True labels')

plt.tight_layout()
plt.show()