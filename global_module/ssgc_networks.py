import torch
from torch import nn
from attontion import PAM_Module, CAM_Module
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish

class Spectral_feature_extraction:
  super(Spectral_feature_extraction, self).__init__()

  def __init__(self, band, classes):
    self.name = SpecFE

    self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
