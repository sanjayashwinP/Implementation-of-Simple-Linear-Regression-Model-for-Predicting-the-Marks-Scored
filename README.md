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
![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/a7f306f2-e01b-4319-b73b-c7e3c4bd955a)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/7026015c-125e-43a9-8f14-9a803cb88721)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/40f86814-9c80-4aa3-a7d1-98bb85bf9fdd)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/a9aad654-4116-492c-ad7e-9bb4f534e4d1)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/c2d2b00e-cbc5-4b76-a188-a3e2199aef01)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/d34ae09e-be02-43a3-83e8-d248df84c8c9)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/54b01f15-fefc-4aa2-afec-85a5eb44cdb5)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/53dda9d4-66ba-4247-9a3b-093c6028f785)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/fe06ac0d-106f-4d33-8f95-3930494ad5d7)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/16784b3c-0588-47be-b764-32de2702a8bf)

![image](https://github.com/sanjayashwinP/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473265/2e59e423-d4c9-4659-98ca-b0b414ef283a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
