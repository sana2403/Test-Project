# -*- coding: utf-8 -*-
"""PETROL CONSUMPTION (DESICION TREE REGRESSER)"""

# Commented out IPython magic to ensure Python compatibility.
# Importing the libraries
import numpy as np  # for array operations
import pandas as pd  # for working with DataFrames
import requests, io  # for HTTP requests and I/O commands
import matplotlib.pyplot as plt  # for data visualization
# %matplotlib inline

# scikit-learn modules
from sklearn.model_selection import train_test_split  # for splitting the data
from sklearn.metrics import mean_squared_error  # for calculating the cost function
from sklearn.tree import DecisionTreeRegressor  # for building the model

# Reading the data
dataset = pd.read_csv("petrol_consumption.csv")
dataset.head()

x = dataset.drop('Petrol_Consumption', axis=1)  # Features
y = dataset['Petrol_Consumption']  # Target

# Splitting the dataset into training and testing set (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)

# Initializing the Decision Tree Regression model
model = DecisionTreeRegressor(random_state = 0)

# Fitting the Decision Tree Regression model to the data
model.fit(x_train, y_train)

# Predicting the target values of the test set
y_pred = model.predict(x_test)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

from sklearn.tree import export_graphviz

# export the decision tree model to a tree_structure.dot file
# paste the contents of the file to webgraphviz.com
export_graphviz(model, feature_names =['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)'])
