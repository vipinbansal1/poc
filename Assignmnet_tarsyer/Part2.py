# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:35:01 2020

@author: GUR45397
"""
import pandas as pd
import numpy as np
import math

df = pd.read_csv("sample1.csv")
feature1 = df['F1']
feature2 = df['F2']
label = df['label']

batch_size = 3
weight1 = 0
weight2 = 0
bias = 0
epoch = 5000
lr = 0.0001

def sigmoidalFunction(func):
    exp = np.exp(-func)
    denom = np.add(exp,1)
    return np.divide(1,denom)

def lossfunction(sigmoidalOutput, labels, starting_index):
    totalLoss = 0
    for index, y in enumerate(labels):
        if(1 == y):
            logVal = math.log(sigmoidalOutput[starting_index+index])
        else:  
            logVal = math.log(1-sigmoidalOutput[starting_index+index])
        totalLoss += logVal
    return -totalLoss/len(labels)

def gradientDescentFactor(sigmoidalOutput, labels, feature):
    step1 = np.subtract(sigmoidalOutput, labels)
    step2 = np.multiply(step1,feature)
    return np.sum(step2)/len(labels)
            
def batch_generator(f1,f2,label,batch_size):
    batch_number = 0
    while(batch_number < 4):
        starting_index = batch_number*batch_size
        ending_index = starting_index+batch_size
        yield(f1[starting_index:ending_index],f2[starting_index:ending_index],
              label[starting_index:ending_index])
        batch_number += 1

for i in range(epoch):
    totalLoss = 0
    for index, (f1,f2,labels) in enumerate(batch_generator(feature1, feature2,
                                          label, batch_size)):
        func = np.multiply(f1, weight1) + np.multiply(f2, weight2) + bias
        sigmoidalOutput = sigmoidalFunction(func)
        loss = lossfunction(sigmoidalOutput, labels, index*batch_size)
        totalLoss =   (totalLoss+loss)/2
        weight1 = weight1 - lr*gradientDescentFactor(sigmoidalOutput, labels, f1)
        weight2 = weight2 - lr*gradientDescentFactor(sigmoidalOutput, labels, f2)
        bias = bias - lr*gradientDescentFactor(sigmoidalOutput, labels, 1)
    print(totalLoss)
    totalLoss = 0
print(weight1,weight2,bias)        




        
