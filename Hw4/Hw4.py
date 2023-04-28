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

X_train_scaled = (X_train - min_vals) / (max_vals - min_vals)

## use mins and max's of columns from training set to scale test set
X_test_scaled = (X_test - min_vals) / (max_vals - min_vals)

### Task 2

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
trainset = dataset(X_train_scaled, y_train)
testset = dataset(X_test_scaled, y_test)

## create their data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

print(X_train_scaled.shape)
print(X_train_scaled.shape[1])

## Deinfe Support Vector Machine (SVM) TODO: this is a fully connected NN
# class SVM(nn.Module):
#     def __init__(self, n_input_features):
#         super(SVM, self).__init__()
#         self.fc1 = nn.Linear(n_input_features, 32)
#         self.fc2 = nn.Linear(32, 64)
#         self.fc3 = nn.Linear(64,1)
#         torch.nn.lin

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class L_SVM(nn.Module):
    def __init__(self, n_input_features):
        super(L_SVM, self).__init__()
        self.fc = nn.Linear(n_input_features, 1)

    def forward(self, x):
        x = self.fc(x)  # SVM should output raw score, no activation function
        return x

## Define logistic regression model
class LRM(nn.Module):
    def __init__(self, n_input_features):
        super(LRM, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
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

    acc_test /= X_test_scaled.shape[0]
    return acc_test
    

## Learning rate and epochs for all models
learning_rate = 0.01
epochs = 500

## Train SVM
modelSVM = L_SVM(n_input_features=X_train_scaled.shape[1])
optimizerSVM = torch.optim.SGD(modelSVM.parameters(),  # best params w grad. desc.
                            lr=learning_rate, 
                            weight_decay=.001) # for L^2 regularization
loss_fnSVM = nn.HingeEmbeddingLoss()   # SVM's minimize sum of hinge losses on train 
## TODO: slides example uses BCEWithLogitsLoss for loss_fn - is that why the test/train there looks like that? Should mine be different?
## see https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be for LRM train/test - very different and my class is modeled off of that

print("training SVM")
trainModel(epochs, trainloader, loss_fnSVM, optimizerSVM, modelSVM)

## Test SVM
print("Accuracy on the test set w/ SVM is: {}".format(testModel(testloader, modelSVM)))

## Train LRM
modelLRM = LRM(n_input_features=X_train_scaled.shape[1])
## According to lecture, LRM is also best w grad. desc and L^2
optimizerLRM = torch.optim.SGD(modelLRM.parameters(),  # best params w grad. desc.
                            lr=learning_rate, 
                            weight_decay=.001) # for L^2 regularization
## However, different loss function
loss_fnLRM = nn.BCEWithLogitsLoss()

print("training LRM")
trainModel(epochs, trainloader, loss_fnLRM, optimizerLRM, modelLRM)

## Test LRM
print("Accuracy on the test set w/ LRM is: {}".format(testModel(testloader, modelLRM)))

## TODO: (see lec) NN's dont perform any better than traditional models on linearly seperable data - for writeup