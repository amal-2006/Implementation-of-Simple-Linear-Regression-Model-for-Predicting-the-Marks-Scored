# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AMALJOSH MAADHAV J
RegisterNumber: 212223230012
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
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
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```
## Output:

### To read head of the file

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/02612c30-75e9-423e-b597-f2c1304fe747)

### To read tail of the file

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/a67565c6-fd64-46d4-b365-9def987ae9c7)

### Compare Dataset

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/86b800f7-5b6f-46b6-b6ca-df1b8816cc48)

### Predicted Value

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/dccb3548-7ef9-497c-a31a-956831b10c3b)

### Graph for Training Dataset

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/40a86ad4-0d31-4b53-adb5-e6792ffafb43)

### Graph for testing Dataset

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/f365668d-d69d-4c67-8221-0eb52cf944d4)

### Error

![image](https://github.com/amal-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/148410730/29786026-97ca-48ee-a459-a73ca17fc321)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
