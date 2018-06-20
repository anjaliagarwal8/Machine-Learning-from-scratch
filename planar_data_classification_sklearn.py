import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

#loading the dataset and visualizing it
X,y = datasets.make_moons(400,noise=0.5)

#plt.scatter(X[:,0],X[:,1],c=Y)
#plt.show()

X,Y = X.T,np.reshape(y,(1,400))

#defining the number of units in each layer
def layer_size(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x,n_y

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#randomly initializing the weights and bias
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    return parameters

#Forward propagation function
def forward_propagate(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    f_parameters = {"Z1" : Z1,
                    "A1" : A1,
                    "Z2" : Z2,
                    "A2" : A2}
    return A2,f_parameters

#The cost function for gradient descent
def cost(X,Y,A2):
    m = X.shape[1]
    costs = -(1/m)*np.sum((np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)))
    costs = np.squeeze(costs)
    return costs

#Backward propagation
def backward_propagate(X,Y,parameters,f_parameters):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1 = f_parameters["A1"]
    A2 = f_parameters["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis = 1,keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    b_parameters = {"dW1" : dW1,
                    "db1" : db1,
                    "dW2" : dW2,
                    "db2" : db2}
    return b_parameters

#Gradient Descent
def gradient_descent(parameters,b_parameters,l_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = b_parameters["dW1"]
    db1 = b_parameters["db1"]
    dW2 = b_parameters["dW2"]
    db2 = b_parameters["db2"]
    
    W1 = W1 - l_rate*dW1
    b1 = b1 - l_rate*db1
    W2 = W2 - l_rate*dW2
    b2 = b2 - l_rate*db2
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    return parameters

#Prediction function
def predict(X,parameters):
    
    A2,f_parameters = forward_propagate(X,parameters)
    prediction = A2>0.5
    
    return prediction

#The NN model
def model(X,Y,n_hidden,l_rate,iterations):
    n_x,n_y = layer_size(X,Y)
    
    parameters = initialize_parameters(n_x,n_hidden,n_y)
    
    for i in range(0,iterations):
        
        A2,f_parameters = forward_propagate(X,parameters)
        costs = cost(X,Y,A2)
        b_parameters = backward_propagate(X,Y,parameters,f_parameters)
        parameters = gradient_descent(parameters,b_parameters,l_rate)
        
        #Printing cost after every 100 iterations
        if i%100 == 0:
            print("The cost at %i iteration is: %f" %(i,costs))
        
    return parameters

#Decision boundary function
def decision_boundary(X,y,model):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    
    
parameters = model(X,Y,30,1.5,2000)
predictions = predict(X,parameters)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

#Plotting the decision boundary
decision_boundary(X, y, lambda x: predict(x.T, parameters))

        
    
    
    