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

![image](https://github.com/user-attachments/assets/266ea771-a3b2-4f74-96af-5c2b70c11dda)

![Screenshot 2024-10-03 140208](https://github.com/user-attachments/assets/7df57952-81fa-49da-afb5-c7aea9896ad3)
![Screenshot 2024-10-03 140300](https://github.com/user-attachments/assets/aa945c02-42b3-4b48-ad20-8915c96770ee)
![Screenshot 2024-10-03 140305](https://github.com/user-attachments/assets/397965d2-df14-4524-b1c5-395e8588e543)
![Screenshot 2024-10-03 140312](https://github.com/user-attachments/assets/0b3bc37b-6d4a-47b6-9f53-8103fc2a2fbf)
![Screenshot 2024-10-03 140330](https://github.com/user-attachments/assets/fbe82156-e6ea-43c6-999b-bba02e3c2cc7)
![Screenshot 2024-10-03 140336](https://github.com/user-attachments/assets/a4356698-ea7c-4f77-b964-14d0613df321)
![Screenshot 2024-10-03 140348](https://github.com/user-attachments/assets/7f4061ab-d426-4d93-bc01-b8477064f06d)
![Screenshot 2024-10-03 140356](https://github.com/user-attachments/assets/5fcc5f2d-6180-46a7-8f10-2f72ace3476f)
![Screenshot 2024-10-03 140402](https://github.com/user-attachments/assets/1b9c8796-b7ac-4ddf-ae5a-114535f0a6bf)




## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
