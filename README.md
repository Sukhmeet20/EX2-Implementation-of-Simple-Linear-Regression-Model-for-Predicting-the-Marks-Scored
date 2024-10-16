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

![Screenshot 2024-10-03 140208](https://github.com/user-attachments/assets/32a1a6ab-88db-4339-8528-d65ebfdf872e)
![Screenshot 2024-10-03 140300](https://github.com/user-attachments/assets/2771dfb0-89aa-450c-b7ad-7b14e1755b07)
![Screenshot 2024-10-03 140305](https://github.com/user-attachments/assets/a50682c7-acd1-4774-83a4-c9a6c5b88d55)
![Screenshot 2024-10-03 140312](https://github.com/user-attachments/assets/ffb9b46c-9f54-4072-9605-eb156897772a)
![Screenshot 2024-10-03 140330](https://github.com/user-attachments/assets/cd59b3a8-3511-4bd8-8a45-3aa2d13dd4a7)
![Screenshot 2024-10-03 140336](https://github.com/user-attachments/assets/8a335487-7c19-42ac-86a3-36463dc67792)
![Screenshot 2024-10-03 140348](https://github.com/user-attachments/assets/9a2664e2-3ea0-44e0-9b3e-b7e06cb3e90d)
![Screenshot 2024-10-03 140356](https://github.com/user-attachments/assets/f750544f-4444-42cb-a2fa-0c9af03b35c5)
![Screenshot 2024-10-03 140402](https://github.com/user-attachments/assets/3c814872-913a-4c68-a2dd-7d02d1caaccb)










## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
