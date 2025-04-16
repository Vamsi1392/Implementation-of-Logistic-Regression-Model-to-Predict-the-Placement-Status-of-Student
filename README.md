# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:M.V.vamsidhar Reddy
RegisterNumber:212224040205 
```
```PY
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### TOP 5 ELEMENTS
![image](https://github.com/user-attachments/assets/a1fdf31b-72c3-4ffa-8582-c98a2c15499f)
![image](https://github.com/user-attachments/assets/c400ae1b-f53b-4b31-9c22-6355d991f6de)
![image](https://github.com/user-attachments/assets/561e09c5-9943-4166-bf8c-3d757db89203)
### DUPLICATE DATA
![image](https://github.com/user-attachments/assets/21d5dc97-b6f9-48c2-8250-f18849a177cf)
### PRINT DATA
![image](https://github.com/user-attachments/assets/4fb64e1b-3e4b-4184-94fb-b3dd71a822db)
### DATA STATUS
![image](https://github.com/user-attachments/assets/5a49cacf-1b92-43a8-bdc9-e4c50cf52d27)
### Y_Prediction array
![image](https://github.com/user-attachments/assets/24837020-925a-409e-9353-95cf1210ae5d)
### CONFUSION ARRAY
![image](https://github.com/user-attachments/assets/28a1aa74-d82a-4e20-b849-6286f032deaa)
### Accuracy value
![image](https://github.com/user-attachments/assets/5df167d4-60fb-4399-b51b-ed607b8b2905)
### Classification report
![image](https://github.com/user-attachments/assets/c1648b2c-7b7b-48de-a8e2-c9ef919d14c1)
### Prediction of LR
![image](https://github.com/user-attachments/assets/011978b2-1db1-47b0-bf24-686da2b54931)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
