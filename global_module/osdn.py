# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
import torch
from torch import nn


class PSA_Channel(nn.Module):
    def __init__(self, inplanes):
        super(PSA_Channel, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = inplanes // 2
        ratio = 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=(1, 1), stride=(1, 1), padding=0,
                                      bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=(1, 1)),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.inplanes, kernel_size=(1, 1))
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def Channel_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)

        context_mask = self.conv_q_right(x)

        context_mask = context_mask.view(batch, 1, height * width)

        context_mask = self.softmax_right(context_mask)
        context_mask = context_mask.transpose(1, 2)
        context = torch.matmul(input_x, context_mask)

        context = context.unsqueeze(-1)

        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def forward(self, x):
        out = self.Channel_pool(x)

        return out


class PSA_Spatial(nn.Module):
    def __init__(self, inplanes):
        super(PSA_Spatial, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = inplanes // 2

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=(1, 1), stride=(1, 1), padding=0,
                                     bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=(1, 1), stride=(1, 1), padding=0,
                                     bias=False)
        self.softmax_left = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def Spatial_pool(self, x):
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        theta_x = self.softmax_left(theta_x)

        context = torch.matmul(avg_x, theta_x)

        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        out = self.Spatial_pool(x)

        return out


class Oneshot_network(nn.Module):
    def __init__(self, bands, classes):
        super(Oneshot_network, self).__init__()

        self.name = 'Oneshot_network'
        inter_bands = ((bands - 7) // 2) + 1
        # initial layer
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        # spectral branch
        self.layer1_1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_4 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_5 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_6 = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=24, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_7 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, inter_bands), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        # spectral attention

        self.attention_spectral = PSA_Channel(24)

        # spatial branch

        self.layer2_0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, bands), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_4 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_5 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_6 = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=24, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        # spatial attention
        self.attention_spatial = PSA_Spatial(24)

        # Classification
        self.GB = nn.Sequential(
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(48, classes)
        )

    def forward(self, x):
        x_01 = self.layer0(x)

        # spectral stream

        x_11 = self.layer1_1(x_01)

        x_12 = self.layer1_2(x_11)

        x_13 = self.layer1_3(x_12)

        x_14 = self.layer1_4(x_13)

        x_15 = self.layer1_5(x_14)

        x_16 = torch.cat((x_11, x_12, x_13, x_14, x_15), dim=1)

        x_17 = self.layer1_6(x_16)

        x_18 = x_01 + x_17

        x_19 = self.layer1_7(x_18)
        x_19 = torch.squeeze(x_19, dim=4)

        # channel attention
        x1 = self.attention_spectral(x_19)

        del x_01, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19

        # spatial stream
        x_20 = self.layer2_0(x)

        x_21 = self.layer2_1(x_20)

        x_22 = self.layer2_2(x_21)

        x_23 = self.layer2_3(x_22)

        x_24 = self.layer2_4(x_23)

        x_25 = self.layer2_5(x_24)

        x_26 = torch.cat((x_21, x_22, x_23, x_24, x_25), dim=1)

        x_27 = self.layer2_6(x_26)

        x_28 = x_20 + x_27
        x_28 = torch.squeeze(x_28, dim=4)
        # spatial attention
        x2 = self.attention_spatial(x_28)

        del x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28

        # classification
        x1 = self.GB(x1)
        x1 = x1.squeeze(-1).squeeze(-1)

        x2 = self.GB(x2)
        x2 = x2.squeeze(-1).squeeze(-1)

        x3 = torch.cat((x1, x2), dim=1)

        output = self.full_connection(x3)

        del x, x1, x2, x3

        return output

# from torchsummary import summary
# model = Oneshot_network(103, 9)
# summary(model, (1, 7, 7, 103), device="cpu")
# exit(0)
