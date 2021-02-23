import pandas as pd
import sys
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.contrib.missing import MissingValuesBar
from yellowbrick.target import ClassBalance
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import Rank1D
from yellowbrick.features import Rank2D
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
from sklearn.ensemble import RandomForestClassifier




col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv('input/breast-cancer-wisconsin.data.csv', names=col,header=None)
df['Bare Nuclei'].replace("?", np.NAN, inplace=True)
df = df.dropna()


df['Class'] = df['Class'] / 2 - 1


X = df.drop(['id', 'Class'], axis=1)
X_col = X.columns
y = df['Class']
X = StandardScaler().fit_transform(X.values)
df1 = pd.DataFrame(X, columns=X_col)
X_train, X_test, y_train, y_test = train_test_split(df1, y,
                                                    train_size=0.8,
                                                    random_state=42)
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2, metric='minkowski')
knn.fit(X_train, y_train)


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

    elif train == False:
        print("Test Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))
features = ['Clump Thickness', 'Uniformity of Cell Size',
      'Uniformity of Cell Shape', 'Marginal Adhesion',
      'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
      'Normal Nucleoli', 'Mitoses']
X = df[features].fillna(0)
classes = df.Class.value_counts().keys()

# sns.distplot(a=X['Clump Thickness'], kde=True)
# sns.jointplot(x='Uniformity of Cell Size', y='Uniformity of Cell Shape', data=X)
# sns.pairplot(df, hue='Class', vars=['Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion'])
X = df[features].as_matrix()
y = df.Class.as_matrix()
# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

