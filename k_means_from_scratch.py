import numpy as np
import math
import matplotlib.pyplot as plt

#data points
x1 = [3,6,2,1,3,4,3,7,8,4,5,11,12,67,89,34,56,78,2,5,7,89,45,3,86,56,90,23,20,68,65,77,79]
x2 = [1,3,4,4,54,78,34,56,89,34,54,65,98,23,45,3,12,34,10,45,23,6,8,87,10,9,1,95,67,12,17,8,10]

#randomly initializing the cluster centroids
def centroid_initialize(m):
   c = np.random.randint(0,m)
   return c


#Assigning the cluster to each point
def cluster(x1,y1,x2,y2,x_list,y_list):
    d = []
    for i in range(len(x_list)):
    
        if (math.sqrt((x_list[i]-x1)**2 + (y_list[i] - y1)**2) < math.sqrt((x_list[i] - x2)**2 + (y_list[i] - y2)**2)):
            d.append(1)
        else:
            d.append(2)
        
    return(d)
    
#Finding the new centroids using the mean of points
def mean(d,x1,x2):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    j = 0
    k = 0
    for i in range(len(x1)):
        if d[i] == 1:
            sum1 += x1[i]
            sum2 += x2[i]
            j += 1 
        else:
            sum3 += x1[i]
            sum4 += x2[i]
            k += 1
    x1_new = sum1/j
    y1_new = sum2/j
    x2_new = sum3/k
    y2_new = sum4/k
    return x1_new,y1_new,x2_new,y2_new

c1 = centroid_initialize(len(x1)-1)
c2 = centroid_initialize(len(x1)-1)

d = []
d = cluster(x1[c1],x2[c1],x1[c2],x2[c2],x1,x2)

for i in range(1000):
    #new cluster centroids
    x1_new,y1_new,x2_new,y2_new = mean(d,x1,x2)
    d.clear()
    d = cluster(x1_new,y1_new,x2_new,y2_new,x1,x2)
    

#plt.plot(x1[c1],x2[c1],marker='x',markersize=10,color='red')
#plt.plot(x1[c2],x2[c2],marker='x' ,markersize=10,color='orange')

for i in range(len(x1)):
    if d[i] == 1:
        plt.plot(x1[i],x2[i],marker='o',color='red')
        plt.show()
    else:
        plt.plot(x1[i],x2[i],marker='o',color='orange')
        plt.show()

#plt.plot(x1_new,y1_new,marker='X',color='green')
#plt.plot(x2_new,y2_new,marker='X',color='green')
#plt.show()
  