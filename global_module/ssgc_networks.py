import torch
from torch import nn
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish

class SSGC_network(nn.Module):
    def __init__(self, band, classes):
        super(SSGC_network, self).__init__()

        self.name = "SSGC"
        self.r = 2
        # spectral branch

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )

        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, 70), stride=(1, 1, 1)) # kernel size随数据变化

        # spectral feature enhancement stage
        self.conv16 = nn.Conv2d(60, 1,
                    kernel_size=(1, 1), padding="valid", stride=(1,1))
        
        self.softmax11 = nn.Softmax()

        self.conv17 = nn.Conv2d(60, 60//self.r,
                    kernel_size=(1, 1), padding="valid", stride=(1,1))

        self.layer_norm11 = nn.Sequential(
                                    nn.LayerNorm(60//self.r, eps=0.001, elementwise_affine=True),
                                    nn.ReLU(inplace=True)
        )

        self.conv18 = nn.Conv2d(60//self.r, 60,
                    kernel_size=(1, 1), padding="valid", stride=(1,1))

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001,  affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # spatial branch enhancement
        self.global_pooling21 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.softmax21 = nn.Softmax()

        self.conv25 = nn.Conv2d(25, 25//self.r,
                    kernel_size=(1, 1), padding="valid", stride=(1,1))

        self.layer_norm21 = nn.Sequential(
                                    nn.LayerNorm(25//self.r, eps=0.001, elementwise_affine=True),
                                    nn.ReLU(inplace=True)
        )

        self.conv26 = nn.Conv2d(25//self.r, 25,
                    kernel_size=(1, 1), padding="valid", stride=(1,1))

        #feature fusion classification stage
        self.layer_norm31 = nn.Sequential(
                                    nn.LayerNorm(60, eps=0.001, elementwise_affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.global_pooling31 = nn.AdaptiveAvgPool2d((1,1))

        self.full_connection = nn.Sequential(
            nn.Linear(120, classes),
            nn.Softmax()
        )

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        print('x14', x14.shape)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        print('x14', x14.shape)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        print('x16', x16.shape)  # 7*7*97, 60
        x16 = x16.squeeze(-1)
        print('x16', x16.shape)

        #subbranch 1
        x17 = self.conv16(x16)
        print('x17', x17.shape)
        x17 = x17.reshape(x17.shape[0],x17.shape[2]*x17.shape[3],1,1)
        x17 = self.softmax11(x17)
        print('x17', x17.shape)

        #subbranch 2
        x18 = x16.view(x16.shape[0], 60, x16.shape[2] *x16.shape[3])
        print('x18', x18.shape)

        #multiplying both branches
        x17 = x17.reshape(x17.shape[0],1,1, x17.shape[1])
        x18 = x18.reshape(x18.shape[0],1, x18.shape[2], x18.shape[1])
        x19 = torch.matmul(x17, x18)
        print('x19', x19.shape)
        x19 = x19.reshape(x19.shape[0], x19.shape[3], 1, 1)
        print('x19', x19.shape)

        x19 = self.conv17(x19)
        print('x19', x19.shape)

        x19 = x19.view(x19.shape[0], x19.shape[1])
        x19 = self.layer_norm11(x19)
        print('x19', x19.shape)

        x19 = x19.view(x19.shape[0], x19.shape[1],1,1)
        x19 = self.conv18(x19)
        print('x19', x19.shape)

        # adding both initial input and multiplication of the two branches
        x20 = x19 + x16
        print('x20', x20.shape)

        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        print('x25', x25.shape)
        x25 = x25.squeeze(-1)
        print('x25', x25.shape)

        #subbranch 1
        x26 = self.global_pooling21(x25)
        x26 = torch.mean(x26, dim=(2, 3), keepdim=True)
        x26 = self.softmax21(x26)
        print('x26', x26.shape)

        #subbranch 2
        x27 = x25.reshape(x25.shape[0], x25.shape[1], x25.shape[3]*x25.shape[2])
        

        #mat mul of both branches
        x28 = torch.mul(x26, x27)
        x28 = self.conv25(x28)
        x28 = self.layer_norm21(x28)
        x28 = self.conv26(x28)

        x29 = x28.reshape(math.sqrt(x28.shape[2]), math.sqrt(x28.shape[2]), 1)

        #adding input and multiplication of the two branches
        x30 = x29 + x25

        # feature fusion

        x20 = x20.permute(2,0,1)
        x20 = self.global_pooling31(x20)
        x20 = x20.permute(1,2,0)
        x20 = x20.view(1,60)

        x30 = x30.permute(2,0,1)
        x30 = self.global_pooling31(x30)
        x30 = x30.permute(1,2,0)
        x30 = x30.view(1,60)

        x41 = torch.cat((x30, x20))
        output = self.full_connection(x41)

        return output
