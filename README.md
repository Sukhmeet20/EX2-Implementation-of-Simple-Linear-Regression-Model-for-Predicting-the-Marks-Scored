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
Developed by: 
RegisterNumber:  
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
![Screenshot 2024-10-03 140208](https://github.com/user-attachments/assets/91050399-0648-4f4c-89c1-342856fbc7e1)
![Screenshot 2024-10-03 140300](https://github.com/user-attachments/assets/811b6578-2cec-49ea-9ef6-7a683b7401e0)
![Screenshot 2024-10-03 140305](https://github.com/user-attachments/assets/b955828b-1ed1-484c-86df-52cc182bbf59)
![Screenshot 2024-10-03 140312](https://github.com/user-attachments/assets/b6db5959-7393-45a7-89fc-87e94e2861d8)
![Screenshot 2024-10-03 140330](https://github.com/user-attachments/assets/c9e22d79-f929-429c-ab86-ced3837bbb80)
![Screenshot 2024-10-03 140336](https://github.com/user-attachments/assets/03ce0e9e-c6db-4dc1-bebf-52fde9e24d5d)
![Screenshot 2024-10-03 140348](https://github.com/user-attachments/assets/6080e265-f46e-4c24-9ff9-e937498682cd)
![Screenshot 2024-10-03 140356](https://github.com/user-attachments/assets/37f07f18-3922-4959-bd72-68c804e02e54)
![Screenshot 2024-10-03 140402](https://github.com/user-attachments/assets/b5d4ab1e-99fb-4de8-bcc7-380629d734fb)




## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
