#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:45:22 2019

@author: zuha
"""

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        #(3x224x224)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.norm4 = nn.BatchNorm1d(500)


        self.fc2 = nn.Linear(500, 500)
        self.norm5 = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, 133)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        ## Define forward behavior
        x= self.norm1(self.pool(F.relu(self.conv1(x))))
        x= self.norm2(self.pool(F.relu(self.conv2(x))))
        x= self.norm3(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = self.norm4(F.relu(self.fc1(x)))
        x = self.norm5(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        

        
        return x
