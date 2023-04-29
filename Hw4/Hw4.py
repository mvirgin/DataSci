### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 5 May 2023

#### Homework 4

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.metrics import accuracy_score

### Task 1

## Import data
data = np.loadtxt("spambase.data", delimiter = ',')

## split into train and test sets
X = data[:, 0:57]
y = data[:, 57:58]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

## use min-max scaling to scale features in training set between 0 and 1
min_vals = np.amin(X_train, axis = 0)
max_vals = np.amax(X_train, axis = 0)

X_train = (X_train - min_vals) / (max_vals - min_vals)

## use mins and max's of columns from training set to scale test set
X_test = (X_test - min_vals) / (max_vals - min_vals)

### Task 2

## Linear SVM
## Train
svm_clf = LinearSVC(C=1e4, loss='hinge', max_iter=int(1e5))
svm_clf.fit(X_train, y_train.ravel())
## Predict / test set accuracy
svm_y_pred = svm_clf.predict(X_test)
print('Accuracy on test set w/ SVM:', np.mean(svm_y_pred==y_test))

## Logistic Regression
## Train
lrm_clf = linear_model.LogisticRegression(penalty='l2', C=1.0)
lrm_clf.fit(X_train, y_train.ravel())
## Predict / test set accuracy
## column 0 is predicted probabilities of email not being spam
## column 1 is of email being spam (in test dataset)
proba_pred = lrm_clf.predict_proba(X_test)[:, 1]
lrm_y_pred = proba_pred > 0.5   # convert to binaray predictions
print('Accuracy on test set w/ LRM:', accuracy_score(y_test, lrm_y_pred))

## TODO: examine coefficients for writeup
# Again, we can access the parameters of a trained classifier using:
# print(clf.coef_)
# print(clf.intercept_)

### Task 3

## Define dataset class
class dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.length

## create training and test datasets
trainset = dataset(X_train, y_train)
testset = dataset(X_test, y_test)

## create their data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

## define neural network 1

class Net1(nn.Module):
    def __init__(self, n_input_features):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(n_input_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)         # note that sigmoid is not included
        return x
    
class Net2(nn.Module):
    def __init__(self, n_input_features):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(n_input_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)        
        x = self.sigmoid(x)
        return x
    
class Net3(nn.Module):
    def __init__(self, n_input_features):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(n_input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

## define function for training a model TODO: assuming they can all be trained the same way?
## takes # of epochs, a trainloader, a loss function, an optimizer, and a model
## trains model using the loss function and optimizer given on the data
## in the trainloader over the provided # of epochs
## returns nothing, prints the loss of the last batch of each epoch
def trainModel(epochs, trainloader, loss_fn, optimizer, model): # I can use these for my fully connected neural network
    for epoch in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(trainloader):
            logits_batch=model(X_batch)
            loss = loss_fn(logits_batch, y_batch.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print("Last batch of epoch {} has loss {}".format(epoch, loss))

## takes testloader and a model, evaluates the accuracy of that model on the
## data held in the testloader. Returns the accuracy of the evalutation
def testModel(testloader, model):
    acc_test = 0.0
    for test_batch_idx, (X_test_batch, y_test_batch) in enumerate(testloader):
        # compute network output (class scores)
        logits_test_batch = model(X_test_batch)
        # accuracy on test batch
        pred_probas_test_batch = torch.sigmoid(logits_test_batch)
        pred_test_batch = pred_probas_test_batch.reshape(-1).detach().numpy().round()
        y_test_batch = y_test_batch.detach().numpy()
        acc_test += (pred_test_batch == y_test_batch).sum()

    acc_test /= X_test.shape[0]
    return acc_test
    
## Learning rate and epochs for all models
learning_rate = 0.01
epochs = 500

## Define and train each model
model1 = Net1(n_input_features=X_train.shape[1])
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, weight_decay=0.001)

model2 = Net2(n_input_features=X_train.shape[1])
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate, weight_decay=0.001)

model3 = Net3(n_input_features=X_train.shape[1])
optimizer3 = torch.optim.SGD(model3.parameters(), lr=learning_rate, weight_decay=0.001)

loss_fn = torch.nn.BCEWithLogitsLoss() # sigmoid is included here

print("\nTraining model 1 \n")
trainModel(epochs, trainloader, loss_fn, optimizer1, model1)
print("\nTraining model 2 \n")
trainModel(epochs, trainloader, loss_fn, optimizer2, model2)
print("\nTraining model 3 \n")
trainModel(epochs, trainloader, loss_fn, optimizer3, model3)

## Test each model
print("\nmodel 1 acc. on test set: {}".format(testModel(testloader, model1)))
print("model 2 acc. on test set: {}".format(testModel(testloader, model2)))
print("model 3 acc. on test set: {}".format(testModel(testloader, model3)))

## TODO: (see lec) NN's dont perform any better than traditional models on linearly seperable data - for writeup