# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:14:57 2022

@author: Florian Martin

Pneumnonia detection using CNNs

Input shape = (150, 150, 1)

"""

import torch.nn as nn
import torch.nn.functional as F


class CNN (nn.Module):
    
    def __init__(self) :
        
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding = 1)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding = 1)
        
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding = 1)
        
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding = 1)
        
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(kernel_size = (2, 2), stride = 2, padding = 1)
        
        
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        
    
    def forward(self, x) :
        x = F.relu(self.conv1)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.drop1(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.drop2(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.drop2(x)
        x = self.bn5(x)
        x = self.pool5(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1)
        x = self.drop2(x)
        x = F.sigmoid(self.fc2(x))
        
        self.initialize_weights()
        
        
    def initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, a = 0.1)
                
                if m.bias is not None :
                    nn.init.constant_(m.bias, 0)
        