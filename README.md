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
![Screenshot 2024-10-17 092347](https://github.com/user-attachments/assets/afbc67d2-798e-42f4-b9ff-763a466024b1)
![Screenshot 2024-10-17 092401](https://github.com/user-attachments/assets/94c6eb7c-460f-49ab-bb29-483fe91a0eed)
![Screenshot 2024-10-17 092413](https://github.com/user-attachments/assets/af989cb2-54e6-47dd-b210-3c69bbd204a2)
![Screenshot 2024-10-17 092421](https://github.com/user-attachments/assets/bedb6d00-9ece-4dda-9a7a-bc5ea134281f)
![Screenshot 2024-10-17 092433](https://github.com/user-attachments/assets/58c2153a-c4db-416e-8720-6ae668145e58)
![Screenshot 2024-10-17 092440](https://github.com/user-attachments/assets/b64a35d3-6489-4326-b9cf-e8d82ae45401)
![Screenshot 2024-10-17 092455](https://github.com/user-attachments/assets/043de9a3-2dc0-4407-804e-f109538740ba)
![Screenshot 2024-10-17 092515](https://github.com/user-attachments/assets/9a13769d-1551-4563-992f-b6690a30ef77)
![Screenshot 2024-10-17 092523](https://github.com/user-attachments/assets/b7122c6f-86c3-409d-bec3-d3a4efad64b9)
## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
