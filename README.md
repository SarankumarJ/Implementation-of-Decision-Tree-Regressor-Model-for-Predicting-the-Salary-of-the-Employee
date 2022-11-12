# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries from python.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply to the model from the dataset.

5.Predict the values of the arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model from the dataset.

7.Predict the values of array

8.Apply it to the new unknown values.

## Program:
```py

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sarankumar J
RegisterNumber: 212221230087

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
![image](https://user-images.githubusercontent.com/94778101/201475868-4dc6a93e-9467-4d74-8d02-52ef4aaae623.png)

![image](https://user-images.githubusercontent.com/94778101/201475875-c6e9ed05-c8e0-4cf5-a26b-518f2c19c09f.png)

![image](https://user-images.githubusercontent.com/94778101/201475878-f0c4bc8b-2932-4a22-b317-567ef11b50fe.png)

![image](https://user-images.githubusercontent.com/94778101/201475885-9d9369fa-8234-419b-9eb3-d03736ecf314.png)

![image](https://user-images.githubusercontent.com/94778101/201475895-d06a412d-6743-4e24-91d3-98b1c3b1826c.png)

![image](https://user-images.githubusercontent.com/94778101/201475908-7df53f15-f88d-4536-95c7-72cd3d3a794a.png)

![image](https://user-images.githubusercontent.com/94778101/201475922-db213d41-5823-4c99-a526-7768bf207abb.png)

![image](https://user-images.githubusercontent.com/94778101/201475964-3870d767-f894-4db3-b181-a41a32b70f76.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
