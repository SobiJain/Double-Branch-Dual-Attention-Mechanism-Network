import torch
import torch.nn as nn

class Mish(torch.nn.Module):
  def __init__(self):
    super(Mish, self).__init__()

  def forward(self, x):
    return x * torch.tanh(torch.log1p(torch.exp(x)))

def calculate_groups(num_filters):
    num_filters = num_filters + 1  # Total number of filters plus one
    return [num_filters // 2, num_filters // 4, num_filters - 1 - (num_filters // 2)]

num_groups = 3

class CustomGroupedConv3d(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, padding="valid", activation=nn.ReLU, strides=(1, 1, 1)):
        super().__init__()
        output_channels = calculate_groups(out_channels)

        self.conv_groups = nn.ModuleList()
        for i in range(num_groups):
            self.conv_groups.append(nn.Conv3d(in_channels=in_channel, out_channels=output_channels[i],
                                              kernel_size=kernel_size, padding=padding,
                                              stride=strides, bias=False))  # Note: bias=False for grouped convolution

    def forward(self, inputs):
        input_channels = inputs.shape[-1]
        input_channel_groups = torch.split(inputs, input_channels // num_groups, dim=1)

        output_channel_groups = []
        for i in range(num_groups):
            output = self.conv_groups[i](input_channel_groups[i])
            output_channel_groups.append(output)

        output = torch.cat(output_channel_groups, dim=1)
        output = self.activation(output)
        return output

def spectral_one_shot_pyramid_network(input_layer):
   """
   Spectral One Shot Pyramid Network (S-OSPN) model in PyTorch.

   Args:
       input_layer (torch.Tensor): Input tensor.

   Returns:
       torch.Tensor: Output tensor.
   """

   # Assuming custom_grouped_conv3d is implemented for PyTorch
   x1 = CustomGroupedConv3d(in_channel=24, out_channels=12, kernel_size=(1, 1, 5), padding="same")(input_layer)
   x2 = CustomGroupedConv3d(in_channel=24, out_channels=12, kernel_size=(1, 1, 3), padding="same")(input_layer)
   x3 = CustomGroupedConv3d(in_channel=24, out_channels=12, kernel_size=(1, 1, 1), padding="same")(input_layer)

   x4 = torch.cat([x1, x2, x3], dim=1)
   x4 = nn.BatchNorm3d(num_features=36, eps=1e-3, momentum=0.1)(x4)
   x4 = Mish()(x4)  # Assuming mish activation is available

   x5 = CustomGroupedConv3d(in_channel=36, out_channels=24, kernel_size=(1, 1, 1), padding="same")(x4)
   x5 = nn.BatchNorm3d(num_features=24, eps=1e-3, momentum=0.1)(x5)
   x5 = Mish()(x5)

   return x5

import torch
import torch.nn as nn

def spectral_one_shot_dense_pyramid_network(input_layer):
    """
    Spectral One Shot Dense Network (S-OSDPN) model.

    Args:
        input_layer: Input layer of the model (tensor).

    Returns:
        Output layer of the block (tensor).
    """

    print('X', input_layer.shape)

    # Convolutional layer 1
    x1 = nn.Conv3d(in_channels=input_layer.shape[1], out_channels=24, kernel_size=(1, 1, 7), stride=(1, 1, 2))(input_layer)
    x1 = nn.BatchNorm3d(num_features=24, eps=0.001, momentum=0.1, track_running_stats=True)(x1)  # Ensure tracking running stats
    x1 = Mish()(x1)

    # Convolutional layers 2-4 with spectral_one_shot_pyramid_network
    x2 = spectral_one_shot_pyramid_network(x1)
    x3 = spectral_one_shot_pyramid_network(x2)
    x4 = spectral_one_shot_pyramid_network(x3)

    # Concatenation and subsequent layers
    x5 = torch.cat([x2, x3, x4], dim=1)
    x6 = nn.Conv3d(in_channels=x5.shape[1], out_channels=24, kernel_size=(1, 1, 1))(x5)
    x6 = nn.BatchNorm3d(num_features=24, eps=0.001, momentum=0.1, track_running_stats=True)(x6)
    x6 = Mish()(x6)
    x6 = x1 + x6  # Perform element-wise addition

    x7 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, 132))(x6)
    x7 = nn.BatchNorm3d(num_features=24, eps=0.001, momentum=0.1, track_running_stats=True)(x7)
    x7 = Mish()(x7)

    w1_shape = x7.shape
    x7 = x7.view(w1_shape[0], w1_shape[1], w1_shape[2], -1)  # Reshape using view

    return x7

r = 2

class Copa(nn.Module):
    def __init__(self):
        super(Copa, self).__init__()
        self.conv1k = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1, padding="same", dilation=4)
        self.conv1q = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1, padding="same", dilation=2)
        self.conv1v = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1, padding="same", dilation=1)
        self.pool = nn.AvgPool2d(kernel_size=(12, 1))
        self.softmax = nn.Softmax(dim=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12//r, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=12//r, out_channels=24, kernel_size=1)
        self.ln = nn.LayerNorm([24, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x1_k = self.conv1k(x)
        x1_q = self.conv1q(x)
        x1_kq = x1_k * x1_q

        x1_v = self.conv1v(x)
        x1_v = x1_v.permute(0, 2, 1, 3)  # Reshape for attention

        x1_p = self.softmax(self.pool(x1_v))

        x3 = torch.matmul(x1_p, x1_kq)

        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = self.ln(x5)
        x5 = self.relu(x5)
        x5 = self.sigmoid(x5)
        x5 = x * x5

        x6 = self.global_pool(x5)
        return x6

def spatial_one_shot_pyramid_network(input_layer):
    """
    Spatial One Shot Pyramid Network (Sa-OSPN) block in PyTorch.

    Args:
        input_layer: Input tensor.

    Returns:
        Output tensor of the block.
    """

    # Convolutional layers with grouped convolutions
    x1 = CustomGroupedConv3d(in_channel=12, out_channels=6, kernel_size=(5, 5, 1), padding="same")(input_layer)
    x2 = CustomGroupedConv3d(in_channel=12, out_channels=6, kernel_size=(3, 3, 1), padding="same")(input_layer)
    x3 = CustomGroupedConv3d(in_channel=12, out_channels=6, kernel_size=(1, 1, 1), padding="same")(input_layer)

    # Concatenation and activation
    x4 = torch.cat([x1, x2, x3], dim=1)  # Concatenation along channel dimension
    x4 = nn.BatchNorm3d(x4.shape[1])(x4)  # Batch normalization
    x4 = Mish()(x4)  # Mish activation

    # Final convolution
    x5 = CustomGroupedConv3d(in_channel=36, out_channels=12, kernel_size=(1, 1, 1), padding="same")(x4)
    x5 = nn.BatchNorm3d(x4.shape[1])(x5)  # Batch normalization
    x5 = Mish()(x5)  # Mish activation

    return x5

def spatial_one_shot_dense_pyramidal_network(input_layer):
    """
    Spatial One Shot Dense Network (Sa-OSDN) model in PyTorch.

    Args:
        input_layer: Input tensor of shape (batch_size, channels, H, W, D).

    Returns:
        Output tensor of the block.
    """

    # Define essential modules for clarity and consistency
    conv3d = nn.Conv3d
    bn = nn.BatchNorm3d
    mish = Mish()  # Assuming Mish activation is available

    # Convolutional layer 1
    x1 = conv3d(in_channels=input_layer.shape[1], out_channels=12, kernel_size=(1, 1, 270))(input_layer)
    x1 = bn(x1)
    x1 = mish(x1)

    # Convolutional layer 2-4 (assuming `spatial_one_shot_pyramid_network` is defined elsewhere)
    x2 = spatial_one_shot_pyramid_network(x1)
    x2 = bn(x2)
    x2 = mish(x2)

    x3 = spatial_one_shot_pyramid_network(x2)
    x3 = bn(x3)
    x3 = mish(x3)

    x4 = spatial_one_shot_pyramid_network(x3)
    x4 = bn(x4)
    x4 = mish(x4)

    # Convolutional layer 5
    x5 = torch.cat([x2, x3, x4], dim=1)  # Concatenate along channel dimension

    # Convolutional layer 6
    x6 = conv3d(in_channels=x5.shape[1], out_channels=12, kernel_size=(1, 1, 1))(x5)
    x6 = bn(x6)
    x6 = mish(x6)
    x6 = x1 + x6  # Element-wise addition (no need for `Add()` module in PyTorch)

    # Reshape
    x7 = x6.view(x6.shape[0], x6.shape[1], x6.shape[2], x6.shape[4])  # Reshape using view()

    return x7

import torch
from torch import nn
from torch.nn import functional as F


class SoPAAtrous(nn.Module):
    def __init__(self, window_size):
        super(SoPAAtrous, self).__init__()
        self.window_size = window_size

        # Replace Conv2D with nn.Conv2d, adjust filters and kernel size
        self.conv1_k = nn.Conv2d(in_channels=input.shape[1], out_channels=6, kernel_size=1, dilation=4, padding="same")
        self.conv1_q = nn.Conv2d(in_channels=input.shape[1], out_channels=6, kernel_size=1, dilation=2, padding="same")
        self.conv1_v = nn.Conv2d(in_channels=input.shape[1], out_channels=6, kernel_size=1, dilation=1, padding="same")

        # Use nn.AdaptiveAvgPool2d instead of GlobalAveragePooling2D for flexibility
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # Print statements removed for cleaner output

        x1_k = self.conv1_k(input)
        x1_q = self.conv1_q(input)

        x1_kq = x1_k * x1_q  # Element-wise multiplication
        x1_kq = x1_kq.view(x1_kq.size(0), -1)  # Reshape

        x1_v = self.conv1_v(input)
        x1_p = self.pool(x1_v)
        x1_p = F.softmax(x1_p, dim=1)  # Softmax along channel dimension
        x1_p = x1_p.view(x1_p.size(0), x1_p.size(1), 1)  # Reshape

        x3 = torch.matmul(x1_kq, x1_p)  # Matrix multiplication
        x3 = x3.view(x3.size(0), self.window_size, self.window_size, 1)  # Reshape

        x4 = F.sigmoid(x3)  # Sigmoid activation

        x5 = x4 * input  # Element-wise multiplication
        x6 = self.pool(x5)

        return x6

class LogOSPN(nn.Module):
    def __init__(self, output_units):
        super(LogOSPN, self).__init__()

        self.output_units = output_units

        self.copa = Copa()
        self.sopa = SoPAAtrous(11)

    def forward(self, input_layer):

        b1 = spectral_one_shot_dense_pyramid_network(input_layer)
        res1 = self.copa(b1)

        b2 = spatial_one_shot_dense_pyramidal_network(input_layer)
        res2 = self.sopa(b2)

        result = torch.cat([res1, res2], dim=1)  # Concatenate along channel dimension

        flatten_layer = result.view(result.size(0), -1)  # Flatten

        output_layer = nn.Linear(flatten_layer.shape[1], output_units)(flatten_layer)  # Dense layer
        output_layer = nn.Softmax()(output_layer)

        return output_layer      
