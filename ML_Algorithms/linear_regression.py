
"""LINEAR_REGRESSION"""
"""1. Load Libraries"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""2. Import Dataset"""

#Importing the dataset
Salary_data = pd.read_csv('Salary_Data.csv')

Salary_data
#type(Salary_data)

#Extracting the dependent and independent variable
X = Salary_data.iloc[:,:-1].values
y = Salary_data.iloc[:,1].values

y

sns.distplot(Salary_data['YearsExperience'],kde=False,bins=10)

sns.countplot(y='YearsExperience',data=Salary_data)

sns.barplot(x='YearsExperience',y='Salary',data=Salary_data)

"""Splitting the dataset into Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

"""Fitting simple Linear regression to the training set"""

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

"""Predicting the test results"""

y_pred = lr.predict(X_test)
y_pred

"""Visualizing the training set results"""

plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title('Salary-Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""[link text](https://)Visualizing the test set results"""

plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title('Salary-Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""Calculating the residuals"""

from sklearn import metrics
print('MAE :',metrics.mean_absolute_error(y_test,y_pred))
print('MSE :',metrics.mean_squared_error(y_test,y_pred))
print('RMSE :',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
