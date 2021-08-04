#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:41:02 2021

@author: slaiad
"""


import torch as tc
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

batchSize = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


trainDataset = datasets.MNIST(root = r'Dataset/MNIST/', train = True, download = True, transform = transform)
print(len(trainDataset))
trainLoader = DataLoader(trainDataset, shuffle = True, batch_size = batchSize, num_workers = 64)

testDataset = datasets.MNIST(root = r'Dataset/MNIST/', train = False, download = True, transform = transform)
testLoader = DataLoader(testDataset, shuffle = True, batch_size = batchSize, num_workers = 64)

class CNN(tc.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tc.nn.Conv2d(1, 10, 5) #10*24*24
        self.pool  = tc.nn.MaxPool2d(2)     #10*12*12
        self.conv2 = tc.nn.Conv2d(10, 20, 5)#20*8*8  and pooling 20*4*4
        self.full  = tc.nn.Linear(320, 10) 
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(batch_size, -1)
        # if x.shape[1] == 160:
        #     print(x.shape)
        x = self.full(x)
        return x
    

def train():
    totalLoss = 0
    for trainIndex, (inputs, labels) in enumerate(trainLoader, 0):
        outputs = cnn(inputs)
        loss = lossCalculator(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()
        
        if trainIndex % 500 ==0:
            print('Training epoch %d, trainIndex %d, loss %4f' % (epoch,  trainIndex, totalLoss))
        
def test():
    correctNum = 0
    totalNum = 0
    
    with tc.no_grad():
        for (inputs, labels) in testLoader:
            outputs = cnn(inputs)
            _, predictedClass = tc.max(outputs.data, dim = 1)#
            correctNum += (predictedClass == labels).sum().item()
            totalNum += labels.size(0)
            
    print('Test accuracy: %2f %%' % (100*correctNum/totalNum))

cnn = CNN()

lossCalculator = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.SGD(cnn.parameters(), lr = 0.1, momentum = 0.5)
if __name__ == '__main__':
    for epoch in range(20):
        train()
        test()
        
        