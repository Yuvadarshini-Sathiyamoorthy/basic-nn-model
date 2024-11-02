# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.


## Neural Network Model

![image](https://github.com/user-attachments/assets/b45ef450-0e95-477f-ae56-e52936adff8e)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

**Name:** Yuvdarshini S

**Register Number:** 212221230126

### Importing Required packages

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
```
### Authenticate the Google sheet and create Dataframe
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open('student_data').sheet1
data=worksheet.get_all_values()

dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'input':float})
dataset1=dataset1.astype({'output':float})
dataset1.head()
X=dataset1[['input']].values
y=dataset1[['output']].values
```
### Split the testing and training data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)

```
### Build the Deep learning Model
```
#Create the model
ai_brain = Sequential([
    Dense(15, activation='relu',input_shape=[1]),
    Dense(9, activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer='rmsprop',loss='mse')

#fit the model
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
print("Yuvadarshini S(212221230126)")
```
### Evaluate the Model
```
loss_df=pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1=Scaler.transform(X_test)

ai_brain.evaluate(X_test1,y_test)

from tensorflow.keras.metrics import RootMeanSquaredError as rmse
err = rmse()
preds = ai_brain.predict(X_test1)
err(y_test,preds)

err = rmse()
print("Name: Yuvadarshini S\n")
print("Root Mean Squred Error:",err(y_test,preds).numpy())

X_n1=[[30]]

X_n1_1=Scaler.transform(X_n1)

print("Name: Yuvadarshini S\n")
ai_brain.predict(X_n1_1)
```



## OUTPUT
## Dataset Information
![image](https://github.com/user-attachments/assets/86eca3a1-9c9a-4c4b-a197-0b37878d6479)


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/e9809b46-b0d6-46c3-834e-81ccba83ceb6)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/eec88c5e-166c-41f7-8799-9aed9fc0f9f2)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/71bd54f5-f9d4-4a92-b911-ae3e288552bf)



## RESULT

Thus a basic neural network regression model for the given dataset is written and executed successfully.
