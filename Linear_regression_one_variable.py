#implement linear regression with one variable to predict profits for a food truck
import numpy as np
import matplotlib.pyplot as plt
import csv

#Reading the dataset
with open("ex1data1.txt","r") as datafile:
    csvreader = csv.reader(datafile)
    population = []
    profit = []
    for row in csvreader:
        population.append(float(row[0]))
        profit.append(float(row[1]))

l_rate = 0.01
b0 = 0
b1 = 0
def activation(b0,b1,x):
    return b0 + b1*x

def cost(b0,b1,x,y):
    sum1 = 0
    for j in range(len(x)):
        sum1 += (activation(b0,b1,x[j]) - y[j])**2
    return(sum1/(2*len(x)))

def gradient_descent(b0,b1,x,y,a):
    sum1 = 0
    sum2 = 0
    for j in range(len(x)):
        sum1 += (activation(b0,b1,x[j])-y[j])*x[j]
        sum2 += (activation(b0,b1,x[j])-y[j])
    b1 = b1 - (a/len(x))*sum1
    b0 = b0 - (a/len(x))*sum2
    return b0,b1

for i in range(1500):
    b0,b1 = gradient_descent(b0,b1,population,profit,l_rate)
print(cost(b0,b1,population,profit))

final = []
for j in range(len(population)):
    final.append(b0+b1*population[j])

plt.plot(population,final)
plt.scatter(population,profit,color="red")
plt.xlabel("population")
plt.ylabel("profit")
plt.show()



        