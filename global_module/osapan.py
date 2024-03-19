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

class SpecAPNBA(nn.Module):
    def __init__(self, h, w, C):
        super(SpecAPNBA, self).__init__()
        
        self.name = 'SpecAPNBA'
        self.C = C
        self.h = h
        self.w = w
        self.r = 2
        self.f = 12

        self.conv11 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 4)
        self.conv21 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 2)
        self.conv31 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 1)

        self.adpt_avg_pooling31 = nn.AdaptiveAvgPool2d(1)
        self.softmax31 = nn.Softmax()

        self.conv41 = nn.Conv2d(self.f, self.f//self.r, kernel_size=(1,1))
        self.conv42 = nn.Conv2d(self.f//self.r, self.C, kernel_size=(1,1))

        self.LN_relu_sig41 = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self,X):

        batch_size = X.shape[0]

        k = self.conv11(X)
        q = self.conv21(X)

        kq = k*q

        kq = kq.reshape(batch_size, self.f, 1, self.h*self.w)
        # -------------------------------------------------------------------------

        v = self.conv31(X)
        v = torch.permute(v, (0, 2, 3, 1))
        v = v.reshape(batch_size, self.f, 1, self.h*self.w)

        v = self.adpt_avg_pooling31(v)
        print('v', v.shape)
        v = self.softmax31(v)

        x41 = torch.einsum("bkij,blik->bijl", kq, v)
        x41 = self.conv41(x41)
        x41 = self.conv42(x41)
        print('x41', x41.shape)

        x42 = self.LN_relu_sig41(x41)
        print('x42', x42.shape)

        output = x42*X
        print('output', output.shape)

        return output

class SpatAPNBA(nn.Module):
    def __init__(self, h, w):
        super(SpatAPNBA, self).__init__()

        self.name = 'SpatAPNBA'
        self.C = 12
        self.h = h
        self.w = w
        self.f = 6

        self.conv11 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 4)
        self.conv21 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 2)
        self.conv31 =  nn.Conv2d(self.C, self.f, kernel_size=(1,1), dilation = 1)

        self.global_pooling21 = nn.AdaptiveAvgPool2d(1)
        self.softmax21 = nn.Softmax()

        self.sigmoid31 = nn.Sigmoid()

    def forward(self, X):

        batch_size = X.shape[0]

        k = self.conv11(X)
        q = self.conv21(X)

        kq = k*q

        kq = kq.reshape(batch_size, self.f, h*w)
        # -------------------------------------------------------------------------

        v = self.conv31(X)

        v = self.global_pooling21(v)
        v = self.softmax21(v)

        x31 = torch.einsum("bik,bjk->bij", kq, v)
        print('x31', x31.shape)

        x32 = x31.reshape(batch_size, self.h, self.w)
        print('x32', x32.shape)
        x32 = self.sigmoid31(x32)

        output = x32*X
        print('output', output.shape)

        return output

class CPCB(nn.Module):
    def __init__(self, h, w, C):
        super(CPCB, self).__init__()

        self.name = 'CPCB'
        self.C = C
        self.h = h
        self.w = w

        self.conv11 = nn.Conv3d(self.C, self.C//4, kernel_size=(1,1,1))
        self.conv21 = nn.Conv3d(self.C, self.C//2, kernel_size=(3,1,1), padding = (1,0,0))
        self.conv31 = nn.Conv3d(self.C, 3*self.C//4, kernel_size=(5,1,1), padding = (2,0,0))

        self.BN_mish = nn.Sequential(
            nn.BatchNorm3d(3*(self.C//2), eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish = nn.Sequential(
            nn.Conv3d(3*(self.C//2), self.C, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.C, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x21 = self.conv21(X)
        x31 = self.conv31(X)
        # print('x11 x21 x31', x11.shape, x21.shape, x31.shape)

        x41 = torch.cat((x11, x21, x31), dim = 1)
        # print('x41', x41.shape)

        x42 = self.BN_mish(x41)
        x43 = self.Conv_BN_mish(x42)
        # print('x43', x43.shape)

        return x43

class SPCB(nn.Module):
    def __init__(self, h, w, C):
        super(SPCB, self).__init__()

        self.name = 'SPCB'
        self.C = C
        self.h = h
        self.w = w

        self.conv11 = nn.Conv3d(self.C, self.C//4, kernel_size=(1,1,1), padding=(0,0,0))
        self.conv21 = nn.Conv3d(self.C, self.C//2, kernel_size=(1,3,3), padding=(0,1,1))
        self.conv31 = nn.Conv3d(self.C, 3*self.C//4, kernel_size=(1,5,5), padding=(0,2,2))

        self.BN_mish = nn.Sequential(
            nn.BatchNorm3d(3*(self.C//2), eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish = nn.Sequential(
            nn.Conv3d(3*(self.C//2), self.C, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.C, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.conv11(X)
        x21 = self.conv21(X)
        x31 = self.conv31(X)
        # print('x11 x21 x31', x11.shape, x21.shape, x31.shape)

        x41 = torch.cat((x11, x21, x31), dim = 1)
        # print('x41', x41.shape)

        x42 = self.BN_mish(x41)
        x43 = self.Conv_BN_mish(x42)
        # print('x43', x43.shape)

        return x43

class SpecFExtraction(nn.Module):
    def __init__(self, h, w, C):
        super(SpecFExtraction, self).__init__()

        self.name = 'SpecFExtraction'
        self.C = C
        self.h = h
        self.w = w
        self.f = 24

        self.Conv_BN_mish11 = nn.Sequential(
            nn.Conv3d(1, self.f, kernel_size=(7, 1, 1), stride = (2,1,1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.cpcb =nn.Sequential(
            CPCB(self.h, self.w, self.f),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish12 = nn.Sequential(
            nn.Conv3d(3*self.f, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish13 = nn.Sequential(
            nn.Conv3d(self.f,self.f, kernel_size=(49, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]
        channels = X.shape[4]

        X = X.reshape(batch_size, 1, channels, self.h, self.w)

        x11 = self.Conv_BN_mish11(X)

        x12 = self.cpcb(x11)
        x13 = self.cpcb(x12)
        x14 = self.cpcb(x13)

        x15 = torch.cat((x12, x13, x14), dim = 1)
        print('x15', x15.shape)

        x16 = self.Conv_BN_mish12(x15)
        print('x16', x16.shape)
        x16 = x11+x16


        x17 = self.Conv_BN_mish13(x16)
        print('x17', x17.shape)
        output = x17.reshape(batch_size, self.h, self.w)
        print('x17', x17.shape)

        return output

class SpatFExtraction(nn.Module):
    def __init__(self, h, w, C):
        super(SpatFExtraction, self).__init__()

        self.name = 'SpatFExtraction'
        self.C = C
        self.h = h
        self.w = w
        self.f = 12

        self.Conv_BN_mish11 = nn.Sequential(
            nn.Conv3d(1, self.f, kernel_size=(103, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.spcb =nn.Sequential(
            SPCB(self.h, self.w, self.f),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish12 = nn.Sequential(
            nn.Conv3d(3*self.f, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]
        channels = X.shape[4]

        X = X.reshape(batch_size, 1, channels, self.h, self.w)

        x11 = self.Conv_BN_mish11(X)

        x12 = self.spcb(x11)
        x13 = self.spcb(x12)
        x14 = self.spcb(x13)

        x15 = torch.cat((x12, x13, x14), dim = 1)
        print('x15', x15.shape)

        x16 = self.Conv_BN_mish12(x15)
        print('x16', x16.shape)
        x16 = x11+x16

        output = x16.reshape(batch_size, self.h, self.w)
        print('output', output.shape)

        return output

class SpecFEnhance(nn.Module):
    def __init__(self, h, w, C):
        super(SpecFEnhance, self).__init__()

        self.name = 'SpecFEnhance'
        self.C = C
        self.h = h
        self.w = w
        self.f = 24

        self.specEnhance = SpecAPNBA(self.h, self.w, self.f)

        self.global_pool_11 = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, X):

        x11 = self.specEnhance(X)
        print("x11", x11.shape)

        x12 = self.global_pool_11(x11)
        print("x12", x12.shape)

        x13 = self.flatten(x13)
        print("x13", x13.shape)

        return x13

class SpatFEnhance(nn.Module):
    def __init__(self, h, w, C):
        super(SpatFEnhance, self).__init__()

        self.name = 'SpatFEnhance'
        self.C = C
        self.h = h
        self.w = w
        self.f = 24

        self.spatEnhance = SpatAPNBA(self.h, self.w)

        self.global_pool_11 = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, X):

        x11 = self.spatEnhance(X)
        print("x11", x11.shape)

        x12 = self.global_pool_11(x11)
        print("x12", x12.shape)

        x13 = self.flatten(x13)
        print("x13", x13.shape)

        return x13


class OSAPAN(nn.Module):
    def __init__(self, bands, classes):
        super(OSAPAN, self).__init__()

        self.name = 'OSAPAN'
        self.f = 24
        self.h = 11
        self.w = 11

        self.SpecFExtraction = SpatFExtraction(self.h, self.w, self.f)

        self.SpecFEnhance = SpecFEnhance(self.h, self.w, self.f)

        self.SpatFExtraction = SpatFExtraction(self.h, self.w, self.f)

        self.SpatFEnhance = SpatFEnhance(self.h, self.w, self.f)

        self.fc = nn.Linear(self.f, classes)

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.SpecFExtraction(X)
        print('x11', x11.shape)

        x11 = self.SpecFEnhance(x11)
        print('x11', x11.shape)

        x12 = self.SpecFExtraction(X)
        print('x12', x12.shape)

        x12 = self.SpecFEnhance(x12)
        print('x12', x12.shape)   

        x13 = torch.cat((x11, x12), dim = 1)
        print('x13', x13.shape) 

        output = self.fc(x13)
        print('output', output.shape) 

        return output
