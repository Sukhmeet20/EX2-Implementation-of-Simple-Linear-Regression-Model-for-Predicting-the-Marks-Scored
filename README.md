# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
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
![Screenshot 2024-10-17 092347](https://github.com/user-attachments/assets/0d2d02ab-5872-436d-870c-2cfa4b23d6e5)
![Screenshot 2024-10-17 092401](https://github.com/user-attachments/assets/87c57526-01a9-4e66-aa7e-f4889973c943)
![Screenshot 2024-10-17 092413](https://github.com/user-attachments/assets/e13666cc-61cd-454b-b7d0-c01e0aa8fc89)
![Screenshot 2024-10-17 092421](https://github.com/user-attachments/assets/0657a605-4059-47ac-a7d9-0ffef1ba7f5a)
![Screenshot 2024-10-17 092433](https://github.com/user-attachments/assets/51b80f6e-170a-4538-a3e5-e3bf38157ee0)
![Screenshot 2024-10-17 092440](https://github.com/user-attachments/assets/a8a9a236-a2cb-4ccc-98f5-c9c5803cb966)
![Screenshot 2024-10-17 092455](https://github.com/user-attachments/assets/5fe58430-6720-47f7-b0d9-fe65a0099a41)
![Screenshot 2024-10-17 092515](https://github.com/user-attachments/assets/fa7bb053-cb8b-4f39-b970-877c1064b68b)
![Screenshot 2024-10-17 092523](https://github.com/user-attachments/assets/e99dca4e-4b2c-47f6-93dc-dbbf34397239)
## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
