import sys
import os
sys.path.append(os.path.abspath('../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F

# simple classifier for discriminating at pixel-level,
# the source from the target, using high-level features
class Classifier(nn.Module):
    def __init__(self, class_num):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(48, class_num, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        
        return x