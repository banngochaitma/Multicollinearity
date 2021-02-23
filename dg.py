import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('data/train.csv')
col_names = ['cycle','setting1','setting2','setting3','s1','s2', 's3', 's4','s5', 's6',  's7', 's8', 's9', 
           's10',  's11', 's12', 's13', 's14', 's15',  's16', 's17', 's18', 's19',  's20', 's21','ttf']
col_name2 = ['cycle','s2', 's3',  's8', 's11',
          's14', 's15',   's17']


from sklearn.model_selection import train_test_split
X = data_train[col_name2]
y = data_train['ttf']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import PredictionError
model = LinearRegression() # Instantiate the linear model and visualizer
visualizer = PredictionError(model=model, identity=False)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()  