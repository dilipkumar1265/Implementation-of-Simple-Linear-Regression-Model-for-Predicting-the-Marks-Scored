# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Developed by: Dilip Kumar R
RegisterNumber:  212222040037
*/

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:

## To read csv file

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/92d7f770-f51d-41cc-b9cf-1aa550e06d63)

## To Read Head and Tail Files

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/aeb9bae5-4e1f-46db-9df2-615ba6045793)

 ## Compare Dataset

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/4b3d10cd-db1c-4c9b-8e7d-4561a0b0e2ed)

## Predicted Value

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/c7976720-7e3b-4796-9d50-14f1bab16204)

## Graph For Training Set

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/014854f4-75fe-4067-b47c-349e84d1cd22)


## Graph For Testing Set

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/71189ba5-4f5b-465f-bcf0-270162c8778e)

## Error

![image](https://github.com/dilipkumar1265/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119065291/9323e23c-229f-459d-9f30-e9261d2d4ab9)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
