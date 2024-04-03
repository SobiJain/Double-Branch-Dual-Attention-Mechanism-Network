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

class Mish(torch.nn.Module):
  def __init__(self):
    super(Mish, self).__init__()

  def forward(self, x):
    return x * torch.tanh(torch.log1p(torch.exp(x)))

def calculate_groups(num_filters):
    num_filters = num_filters + 1  # Total number of filters plus one
    return [num_filters // 2, num_filters // 4, num_filters - 1 - (num_filters // 2 + num_filters // 4)]

num_groups = 3

class CustomGroupedConv3d(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, padding="valid", activation=nn.ReLU, strides=(1, 1, 1)):
        super().__init__()
        output_channels = calculate_groups(out_channels)
        self.out_channels = out_channels

        self.conv_groups = nn.ModuleList()
        for i in range(num_groups):
            self.conv_groups.append(nn.Conv3d(in_channels=in_channel // num_groups, out_channels=output_channels[i],
                                              kernel_size=kernel_size, padding=padding,
                                              stride=strides, bias=False))  # Note: bias=False for grouped convolution

    def forward(self, inputs):
        input_channels = inputs.shape[1]
        input_channel_groups = torch.split(inputs, input_channels // num_groups, dim=1)

        output_channel_groups = []
        for i in range(num_groups):
            output = self.conv_groups[i](input_channel_groups[i])
            output_channel_groups.append(output)

        output = torch.cat(output_channel_groups, dim=1)
        return output

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

        self.adpt_avg_pooling31 = nn.AvgPool2d(kernel_size=(12, 1))
        self.softmax31 = nn.Softmax()

        self.conv41 = nn.Conv2d(self.f, self.f//self.r, kernel_size=(1,1))
        self.conv42 = nn.Conv2d(self.f//self.r, self.C, kernel_size=(1,1))

        self.LN_relu_sig41 = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.ReLU(),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,X):

        batch_size = X.shape[0]

        k = self.conv11(X)
        q = self.conv21(X)

        kq = k*q
        # print('kq', kq.shape)

        kq = kq.reshape(batch_size, self.f, 1, self.h*self.w)
        # print('kq', kq.shape)
        # -------------------------------------------------------------------------

        v = self.conv31(X)
        # print('v', v.shape)
        v = torch.permute(v, (0, 2, 3, 1))
        # print('v', v.shape)
        v = v.reshape(batch_size, self.h*self.w, self.f, 1)
        # print('v', v.shape)

        v = self.adpt_avg_pooling31(v)
        # print('v', v.shape)
        v = self.softmax31(v)

        x41 = torch.einsum("bijk,bkjl->bijl", kq, v)
        # print('x41', x41.shape)

        x41 = self.conv41(x41)
        # print('x41', x41.shape)

        x41 = self.conv42(x41)
        # print('x41', x41.shape)

        x41 = x41.squeeze(-1).squeeze(-1)

        x42 = self.LN_relu_sig41(x41)
        # print('x42', x42.shape)

        x42 = x42.unsqueeze(-1).unsqueeze(-1)

        output = x42*X
        output = self.global_pool(output)
        # print('output', output.shape)

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

        self.global_pooling21 = nn.AdaptiveAvgPool2d((1,1))
        self.softmax21 = nn.Softmax()

        self.sigmoid31 = nn.Sigmoid()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X):

        batch_size = X.shape[0]

        k = self.conv11(X)
        q = self.conv21(X)

        kq = k*q

        kq = kq.reshape(batch_size, self.f, self.h*self.w)
        # print('kq', kq.shape)
        # -------------------------------------------------------------------------

        v = self.conv31(X)
        # print('v', v.shape)
        v = self.global_pooling21(v)
        # print('v', v.shape)
        v = self.softmax21(v)
        v = v.reshape(batch_size, 1, self.f)
        # print('v', v.shape)

        x31 = torch.einsum("bik,bkj->bij", v, kq)
        # print('x31', x31.shape)

        x32 = x31.reshape(batch_size, 1, self.h, self.w)
        # print('x32', x32.shape)
        x32 = self.sigmoid31(x32)

        output = x32*X
        output = self.global_pool(output)
        # print('output', output.shape)

        return output

class CPCB(nn.Module):
    def __init__(self, h, w, C):
        super(CPCB, self).__init__()

        self.name = 'CPCB'
        self.C = C
        self.h = h
        self.w = w

        self.conv11 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(1,1,1), padding = "same")
        self.conv21 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(1,1,3), padding = "same")
        self.conv31 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(1,1,5), padding = "same")

        self.BN_mish = nn.Sequential(
            nn.BatchNorm3d(3*(self.C//2), eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish = nn.Sequential(
            CustomGroupedConv3d(3*(self.C//2), self.C, kernel_size=(1, 1, 1)),
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

        self.conv11 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(1,1,1), padding = "same")
        self.conv21 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(3,3,1), padding = "same")
        self.conv31 = CustomGroupedConv3d(self.C, self.C//2, kernel_size=(5,5,1), padding = "same")

        self.BN_mish = nn.Sequential(
            nn.BatchNorm3d(3*(self.C//2), eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish = nn.Sequential(
            CustomGroupedConv3d(3*(self.C//2), self.C, kernel_size=(1, 1, 1)),
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
            nn.Conv3d(1, self.f, kernel_size=(1, 1, 7), stride = (1,1,2)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.cpcb =nn.Sequential(
            CPCB(self.h, self.w, self.f),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish12 = nn.Sequential(
            nn.Conv3d(2*self.f, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish13 = nn.Sequential(
            nn.Conv3d(self.f,self.f, kernel_size=(1, 1, 85)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]
        channels = X.shape[4]

        x11 = self.Conv_BN_mish11(X)
        # print('x11', x11.shape)

        x12 = self.cpcb(x11)
        # x13 = self.cpcb(x12)
        x14 = self.cpcb(x12)

        x15 = torch.cat((x12, x14), dim = 1)
        # print('x15', x15.shape)

        x16 = self.Conv_BN_mish12(x15)
        # print('x16', x16.shape)
        x16 = x11+x16


        x17 = self.Conv_BN_mish13(x16)
        # print('x17', x17.shape)
        output = x17.reshape(batch_size, self.f, self.h, self.w)
        # print('x17', x17.shape)

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
            nn.Conv3d(1, self.f, kernel_size=(1, 1, 176)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.spcb =nn.Sequential(
            SPCB(self.h, self.w, self.f),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.Conv_BN_mish12 = nn.Sequential(
            nn.Conv3d(2*self.f, self.f, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.f, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

    def forward(self, X):

        batch_size = X.shape[0]
        channels = X.shape[4]

        x11 = self.Conv_BN_mish11(X)
        # print('x11', x11.shape)

        x12 = self.spcb(x11)
        # x13 = self.spcb(x12)
        x14 = self.spcb(x12)

        x15 = torch.cat((x12, x14), dim = 1)
        # print('x15', x15.shape)

        x16 = self.Conv_BN_mish12(x15)
        # print('x16', x16.shape)
        x16 = x11+x16

        output = x16.reshape(batch_size, self.f, self.h, self.w)
        # print('output', output.shape)

        return output

class OSAPAN(nn.Module):
    def __init__(self, bands, classes):
        super(OSAPAN, self).__init__()

        self.name = 'OSAPAN'
        self.f = 24
        self.h = 11
        self.w = 11

        self.SpecFExtraction = SpecFExtraction(self.h, self.w, self.f)

        self.SpecFEnhance = SpecAPNBA(self.h, self.w, self.f)

        self.SpatFExtraction = SpatFExtraction(self.h, self.w, self.f)

        self.SpatFEnhance = SpatAPNBA(self.h, self.w)

        self.fc = nn.Linear(3*self.f//2, classes)
        self.softmax = nn.Softmax()

    def forward(self, X):

        batch_size = X.shape[0]

        x11 = self.SpecFExtraction(X)
        # print('x11', x11.shape)

        x11 = self.SpecFEnhance(x11)
        # print('x11', x11.shape)

        x12 = self.SpatFExtraction(X)
        # print('x12', x12.shape)

        x12 = self.SpatFEnhance(x12)
        # print('x12', x12.shape)   

        x13 = torch.cat((x11, x12), dim = 1)
        x13 = x13.squeeze(-1).squeeze(-1)
        # print('x13', x13.shape) 

        output = self.fc(x13)
        output = self.softmax(output)
        # print('output', output.shape) 

        return output
