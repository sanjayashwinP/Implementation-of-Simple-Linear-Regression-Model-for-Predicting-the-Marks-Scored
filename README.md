# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sanjay Ashwin P
RegisterNumber: 212223040181 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color="purple)
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/555dcf87-edc3-4dde-97d0-099cf222cbfa)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/89c04139-06fb-494c-8182-24b4fed0d4d2)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/31dc9b7f-16ab-4053-a89f-084bdaef9719)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/62246e92-dd4e-47da-b186-0956da0b1dbb)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/82c8aa54-1923-4343-99ce-940a2e0760f0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
