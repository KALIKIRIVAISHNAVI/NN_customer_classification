# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/ce7d30ad-70f1-4fbc-8e92-f7d38161c1ec)

## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.
### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.
### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).
### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.
### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: KALIKIRI VAISHNAVI
### Register Number: 212223040081

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)  # Renamed to fc4 to avoid overwriting

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Changed to fc4 for the final layer
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
    model.train()
    for epoch in range(epochs):
        for X_batch,y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/4bef0f2c-323e-46f9-9c0d-9b893308b898)


## OUTPUT



### Confusion Matrix

![image](https://github.com/user-attachments/assets/19d158fa-659e-451e-8982-76c2e23488d9)



![image](https://github.com/user-attachments/assets/4f313ad6-d998-4327-a719-f59c032a543a)
### Classification Report

![image](https://github.com/user-attachments/assets/51a430d8-f2d3-45e4-82b5-012fb6a87c34)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/c76c03bd-2aaa-4e3b-9050-2d7f274cdfa1)


## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
