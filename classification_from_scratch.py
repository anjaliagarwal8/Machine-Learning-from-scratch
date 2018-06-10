#this is a classification problem in which we have to classifiy cat and non-cat images. We are using logistic regression for classifying

import numpy as np
import matplotlib.pyplot as plt
import h5py

#reading the dataasets
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#Normalizing the input features
def normalize(X):
    X_new = X.reshape(X.shape[0],-1).T
    X_new = X_new/255
    return X_new

#activation function
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

#Defining the parameters to be learnt and the cost function
def propagate(w,b,X,Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*(np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T)))
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    return cost,w,b,dw,db

#Learning the parameters and optimization using Gradient descent
def optimize(w,b,dw,db,X,Y,l_rate,iterations):
    
    costs = []
    for i in range(iterations):
        cost,w,b,dw,db = propagate(w,b,X,Y)
        w = w - l_rate*dw
        b = b - l_rate*db
        if i%100 == 0:
            costs.append(cost)
            print("Cost after %i iteration: %f" %(i,cost))
            
    return w,b,dw,db,costs

#Prediction function for the test set
def predict(w,b,X):
    m = X.shape[1]
    y_predict = np.zeros((1,m))
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(m):
        if A[0,i]>=0.5:
            y_predict[0,i] = 1
        else:
            y_predict[0,i] = 0
    return y_predict

def model(w,b,train_X,train_Y,test_X,test_Y,l_rate,iterations):
        
    cost,w,b,dw,db = propagate(w,b,train_X,train_Y)
    w,b,dw,db,costs = optimize(w,b,dw,db,train_X,train_Y,l_rate,iterations)
    y_predict_train= predict(w,b,train_X)
    y_predict_test = predict(w,b,test_X)
    train_accuracy = 100 - np.mean(np.abs(y_predict_train - train_Y))*100
    test_accuracy = 100 - np.mean(np.abs(y_predict_test - test_Y))*100
    print("Train accuracy:"+str(train_accuracy))
    print("Test accuracy:"+str(test_accuracy))
    return costs
  
train_x_orig,train_y,test_x_orig,test_y,_ = load_dataset()
#print(train_x_orig.shape)
#print(train_y.shape)
#print(test_x_orig.shape)
#print(test_y.shape)

train_x = normalize(train_x_orig)
test_x = normalize(test_x_orig)
#print(train_x.shape)
#print(test_x.shape)

w = np.zeros((train_x.shape[0],1))
b = 0

cost = model(w,b,train_x,train_y,test_x,test_y,0.001,2000)

#plotting the cost function
plt.plot(cost)
plt.show()