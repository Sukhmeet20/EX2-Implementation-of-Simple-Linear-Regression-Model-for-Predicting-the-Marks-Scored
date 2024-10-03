# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: Sukhmeet Kaur G
RegisterNumber: 2305001032
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/ex1.csv')

df.head()
df.tail()

X = df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:

![image](https://github.com/user-attachments/assets/f8f0dbbf-5bf2-42cc-b373-5b013f7027d4)
![image](https://github.com/user-attachments/assets/6dcc79e6-cc59-4d26-bb43-e9738ffdc72a)
![image](https://github.com/user-attachments/assets/9c8e5f1e-4005-4724-9034-8d7e68945d61)
![image](https://github.com/user-attachments/assets/866bc446-fd78-42a4-a89b-280da4c87918)
![image](https://github.com/user-attachments/assets/e6981dd6-edb6-4bf5-988f-8704e0e9464f)
![image](https://github.com/user-attachments/assets/777af76c-6566-4518-b67c-e590bc90fb97)
![image](https://github.com/user-attachments/assets/d39f6d34-3fb5-4058-a569-7b549b7e26f1)
![image](https://github.com/user-attachments/assets/87bc0ffb-9562-4a63-9841-08efabaefc3a)
![image](https://github.com/user-attachments/assets/888337c2-1d19-4946-b4c1-ceffc94180a8)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
