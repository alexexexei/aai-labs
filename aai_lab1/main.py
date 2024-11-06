import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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


def print_acc3(test, res1, res2, res3, name1, name2, name3):
    print(f'accuracy:\n{name1}: {accuracy_score(test, res1):.2f}\
            \n{name2}: {accuracy_score(test, res2):.2f}\
            \n{name3}: {accuracy_score(test, res3):.2f}\n')


def perform(cm1, cm2, cm3, tn, name1, name2, name3):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    sns.heatmap(cm1, annot=True, fmt="d", cmap='viridis', xticklabels=tn, yticklabels=tn, ax=axes[0])
    axes[0].set_title(f'{name1} Confusion Matrix')
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')

    sns.heatmap(cm2, annot=True, fmt="d", cmap='viridis', xticklabels=tn, yticklabels=tn, ax=axes[1])
    axes[1].set_title(f'{name2} Confusion Matrix')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')

    sns.heatmap(cm3, annot=True, fmt="d", cmap='viridis', xticklabels=tn, yticklabels=tn, ax=axes[2])
    axes[2].set_title(f'{name3} Confusion Matrix')
    axes[2].set_xlabel('Predicted labels')
    axes[2].set_ylabel('True labels')

    plt.tight_layout()
    plt.show()


path = os.path.dirname(os.path.abspath(__file__))
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
model = LogisticRegression(max_iter=1000)
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

print_acc3(test_y, yhat, yhat_2, yhat_3, "log_regr", "knn", "rand_forest")
perform(conf_matrix, conf_matrix_2, conf_matrix_3,
        target_names, "LogisticRegression", "KNeighborsClassifier", 
        "RandomForestClassifier")

# model 2.2
yhat_2_2 = KNeighborsClassifier(n_neighbors=2).fit(train_X, train_y).predict(test_X)
conf_matrix_2_2 = confusion_matrix(test_y, yhat_2_2)
# model 2.3
yhat_2_3 = KNeighborsClassifier(n_neighbors=5).fit(train_X, train_y).predict(test_X)
conf_matrix_2_3 = confusion_matrix(test_y, yhat_2_3)

print_acc3(test_y, yhat_2, yhat_2_2, yhat_2_3, "n=2", "n=3", "n=5")
perform(conf_matrix_2, conf_matrix_2_2, conf_matrix_2_3,
        target_names, "KNeighborsClassifier n=2", "KNeighborsClassifier n=3", 
        "KNeighborsClassifier n=5")
