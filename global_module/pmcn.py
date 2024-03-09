import torch
from torch import nn
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F
from activation import mish

import sys
sys.path.append('../global_module/')

class Channel_PSA(nn.Module):
    def __init__(self, h, w):
        
        self.name = 'Channel_PSA'
        self.C = 24
        self.h = h
        self.w = w

        self.conv11 =  nn.Conv2d(C, C//2, kernel_size=(1,1))
        self.conv21 =  nn.Conv2d(C, 1, kernel_size=(1,1))

        self.softmax21 = nn.Softmax()

        self.conv31 = nn.Conv2d(C//2, C, kernel_size=(1,1))
        self.layer_norm31 = nn.LayerNorm(C)
        self.sigmoid31 = nn.Sigmoid()

    def forward(self,X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x11 = x11.reshape(batch_size, C//2, h*w)
        print('x11', x11.shape)

        x21 = self.conv21(X)
        x21 = x21.reshape(batch_size, h*w, 1, 1)
        print('x21', x21.shape)

        x21 = sef.softmax21(x21)

        x31 = torch.einsum("bik,bkjl->bijl", x11, x21)
        print('x31', x31.shape)

        x32 = self.conv31(x31)
        print('x32', x32.shape)
        x32 = self.layer_norm31(x32)
        x32 = self.sigmoid31(x32)

        output = x32*X
        print('output', output.shape)

        return output

class Spatial_PSA(nn.Module):
    def __init__(self, h, w):

        self.name = 'Spatial_PSA'
        self.C = 24
        self.h = h
        self.w = w

        self.conv11 =  nn.Conv2d(C, C//2, kernel_size=(1,1))
        self.conv21 =  nn.Conv2d(C, C//2, kernel_size=(1,1))

        self.global_pooling21 = nn.AvgPool2d(kernel_size=(h, w))

        self.softmax21 = nn.Softmax()
        self.sigmoid31 = nn.Sigmoid()

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x11 = x11.reshape(batch_size, C//2, h*w)
        print('x11', x11.shape)

        x21 = self.conv21(X)
        print('x21', x21.shape)
        x21 = self.global_pooling21(x21)
        print('x21', x21.shape)
        x21 = x21.reshape(batch_size, 1, C//2)
        print('x21', x21.shape)

        x21 = sef.softmax21(x21)

        x31 = torch.einsum("bik,bkj->bij", x21, x11)
        print('x31', x31.shape)

        x32 = x31.reshape(batch_size, 1, h, w)
        print('x32', x32.shape)
        x32 = self.sigmoid31(x32)

        output = x32*X
        print('output', output.shape)

        return output

class Channel_PCB(nn.Module):
    def __init__(self, h, w):

        self.name = 'Channel_PCB'
        self.C = 103
        self.h = h
        self.w = w
        self.f = 24

        self.conv11 = nn.Conv3d(f, f//2, kernel_size=(7,1,1))
        self.conv21 = nn.Conv3d(f, f//2, kernel_size=(5,1,1))
        self.conv31 = nn.Conv3d(f, f//2, kernel_size=(3,1,1))

        self.BN_prelu = nn.Sequential(
            nn.BatchNorm3d(3*(f//2), eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.Conv_BN_prelu = nn.Sequential(
            nn.Conv3d(3*(f//2),f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x21 = self.conv21(X)
        x31 = self.conv31(X)

        x41 = torch.cat((x11, x21, x31), dim = 1)
        print('x41', x41.shape)

        x42 = self.BN_prelu(x41)
        x43 = self.Conv_BN_prelu(x42)
        print('x43', x43.shape)

        return x43

class Spatial_PCB(nn.Module):
    def __init__(self, h, w):

        self.name = 'Channel_PCB'
        self.C = 103
        self.h = h
        self.w = w
        self.f = 24

        self.conv11 = nn.Conv3d(f, f//2, kernel_size=(1,7,7))
        self.conv21 = nn.Conv3d(f, f//2, kernel_size=(1,5,5))
        self.conv31 = nn.Conv3d(f, f//2, kernel_size=(1,3,3))

        self.BN_prelu = nn.Sequential(
            nn.BatchNorm3d(3*(f//2), eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.Conv_BN_prelu = nn.Sequential(
            nn.Conv3d(3*(f//2),f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x21 = self.conv21(X)
        x31 = self.conv31(X)

        x41 = torch.cat((x11, x21, x31), dim = 1)
        print('x41', x41.shape)

        x42 = self.BN_prelu(x41)
        x43 = self.Conv_BN_prelu(x42)
        print('x43', x43.shape)

        return x43

class PMCN(nn.Module):
    def __init__(self, bands, classes):
        super(PMCN, self).__init__()

        self.name = 'PMCN'
        self.f = 24
        self.h = 9
        self.w = 9

        self.Conv_BN_prelu11 = nn.Sequential(
            nn.Conv3d(1, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.channel_pcb = Channel_PCB(self.h, self.w)

        self.Conv_BN_prelu12 = nn.Sequential(
            nn.Conv3d(3*self.f, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.Conv_BN_prelu13 = nn.Sequential(
            nn.Conv3d(self.f,24, kernel_size=(103, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.channel_psa = Channel_PSA(self.h, self.w)

        self.Conv_BN_prelu14 = nn.Sequential(
            nn.Conv3d(self.f,103, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.Conv_BN_prelu21 = nn.Sequential(
            nn.Conv3d(1,24, kernel_size=(103, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.spatial_pcb = Spatial_PCB(self.h,self.w)

        self.Conv_BN_prelu22 = nn.Sequential(
            nn.Conv3d(3*self.f,60, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.spatial_psa = Spatial_PSA(self.h,self.w)

        self.Avg_BN_mish = nn.Sequential(
            nn.AvgPool2d(kernel_size=(15, 15)),
            nn.BatchNorm2d(60),
            mish()
        )

        self.linear = nn.Linear(60, classes)

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.Conv_BN_prelu11(X)
        print('x11', x11.shape)

        x12 = self.channel_pcb(x11)
        print('x12', x12.shape)

        x13 = self.channel_pcb(x12)
        print('x13', x13.shape)

        x14 = self.channel_pcb(x13)
        print('x14', x14.shape)

        x15 = torch.cat((x12, x13, x14), dim = 1)
        x15 = self.Conv_BN_prelu12(x15)
        print('x15', x15.shape)

        x16 = self.Conv_BN_prelu13(x15)
        print('x16', x16.shape)

        x16 = x16.reshape(batch_size, self.f, self.h, self.w)
        print('x16', x16.shape)

        x17 = self.channel_psa(x16)
        print('x17', x17.shape)

        x17 = x16.reshape(batch_size, self.f, 1, self.h, self.w)
        print('x17', x17.shape)

        x18 = self.Conv_BN_prelu14(x17)
        print('x18', x18.shape)

        x18 = x18.reshape(batch_size, x18.shape[2], x18.shape[1], self.h, self.w)
        print('x18', x18.shape)

        x19 = self.Conv_BN_prelu21(x18)
        print('x19', x19.shape)

        x20 = self.spatial_pcb(x19)
        print('x20', x20.shape)

        x21 = self.spatial_pcb(x20)
        print('x20', x21.shape)

        x22 = self.spatial_pcb(x21)
        print('x20', x22.shape)

        x23 = torch.cat((x20, x21, x22), dim = 1)
        print('x23', x23.shape)

        x24 = self.Conv_BN_prelu22(x23)
        print('x24', x24.shape)

        x24 = x24.reshape(batch_size, 60, self.h, self.w)
        print('x24', x24.shape)

        x25 = self.spatial_psa(x24)
        print('x25', x25.shape)

        x25 = self.Avg_BN_mish(x25)
        print('x25', x25.shape)

        x25 = x25.squeeze(-1)
        print('x25', x25.shape)

        output = self.linear()
        print('output', output.shape)

        return output
      
