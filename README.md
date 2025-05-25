# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load tools for data manipulation, encoding, modeling, and evaluation.
2. Load the CSV file into a DataFrame
3. View the first few rows
4. Get an overview of data types and null values.
5. Convert text labels (like “Manager”, “Analyst”, etc.) in the Position column into numeric format.
6. 'x' (features): Independent variables — Position (encoded) and Level
7. 'y' (target): Dependent variable — Salary.
8. Split the dataset into 80% training data and 20% testing data.
9. Create a decision tree model and train it on the training data.
10. Use the trained model to predict salary values for the test set.
11. Measures average of the squared differences between predicted and actual values.
12. R² Score: Indicates how well the model explains variability (closer to 1 = better).
13. Input Explanation: Position = 5 (encoded value) Level = 6

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Meenakshi.R
RegisterNumber: 212224220062
*/
```
```
import pandas as pd

df=pd.read_csv("Salary.csv")
df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["Position"]=le.fit_transform(df["Position"])
df.head()

x = df[["Position", "Level"]]
y = df["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5, 6]])
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
