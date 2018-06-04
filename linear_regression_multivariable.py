#implement linear regression with multiple variables to predict the prices of houses. Suppose you are #selling your house and you want to know what a good market price would be.
#The first column is the size of the house (in square feet), the second column is the number of #bedrooms, #and the third column is the price of the house.
import matplotlib.pyplot as plt
import numpy as np
import csv

#Reading the dataset
with open("ex1data2.txt","r") as datafile:
    csvreader = csv.reader(datafile)
    size = []
    bedrooms = []
    price = []
    for row in csvreader:
        size.append(int(row[0]))
        bedrooms.append(int(row[1]))
        price.append(int(row[2]))
        
#Feature Normalization
def feature_scale(x):
    mean = np.mean(x)
    std_deviation = np.std(x)
    x_new = []
    for i in range(len(x)):
        x_new.append((x[i] - mean)/std_deviation)
    return x_new

#Activation function = b0 + b1x1 + b2x2
def activation(b0,b1,b2,x1,x2):
    return b0 + b1*x1 + b2*x2

def cost(b0,b1,b2,x1,x2,y):
    sum1 = 0
    for j in range(len(x1)):
        sum1 += (activation(b0,b1,b2,x1[j],x2[j]) - y[j])**2
    return(sum1/(2*len(x1)))

def gradient_descent(b0,b1,b2,x1,x2,y,a):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for j in range(len(x1)):
        sum1 += (activation(b0,b1,b2,x1[j],x2[j])-y[j])*x1[j]
        sum3 += (activation(b0,b1,b2,x1[j],x2[j])-y[j])*x2[j]
        sum2 += (activation(b0,b1,b2,x1[j],x2[j])-y[j])
    b1 = b1 - (a/len(x1))*sum1
    b0 = b0 - (a/len(x1))*sum2
    b2 = b2 - (a/len(x1))*sum3
    return b0,b1,b2

size_new = feature_scale(size)
bedrooms_new = feature_scale(bedrooms)
price_new = feature_scale(price)
l_rate = 0.01
b0 = 0
b1 = 0
b2 = 0

cost_f = []
for i in range(1500):
    b0,b1,b2 = gradient_descent(b0,b1,b2,size_new,bedrooms_new,price_new,l_rate)
    cost_f.append(cost(b0,b1,b2,size_new,bedrooms_new,price_new))
    
print(cost(b0,b1,b2,size_new,bedrooms_new,price_new))
#testing for the convergence of cost function
plt.plot(cost_f)
plt.show()

