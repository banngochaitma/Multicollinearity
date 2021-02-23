from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import PredictionError

from sklearn import datasets, linear_model

data_train = pd.read_csv('data/test.csv')
col_sensor = ['cycle','s2', 's3', 's4',  's7', 's8', 's9',
             's11', 's12', 's13', 's14', 's15',  's17',  's20', 's21']

X = data_train[col_sensor]
y = data_train['ttf']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(X, y)
print( 'Solution found by scikit-learn  : ', regr.coef_ )


model = LinearRegression() # Instantiate the linear model and visualizer
visualizer = PredictionError(model=model, identity=False)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()
