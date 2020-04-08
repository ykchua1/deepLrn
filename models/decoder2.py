# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

# This is version 2 of the decoder module.
# Added functionality to output intermediate layer output.
# To output the intermediate layer activations just before the predictions.

import sys
import os
sys.path.append(os.path.abspath('../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from deepLrn.models.mobilenet import MobileNetV2
from deepLrn.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from deepLrn.models.encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(24, 4, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(4, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(54, 48, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(48, class_num, kernel_size=1)

        self._init_weight()



    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4 = F.interpolate(x, size=low_level_feature.size()[2:4], mode='bilinear' ,align_corners=True)
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat_pred = self.bn3(x_4_cat) # from here down, added by ykchua1
        x_4_cat_pred = self.relu(x_4_cat_pred)
        x_4_cat_pred = self.dropout3(x_4_cat_pred)
        x_4_cat_pred = self.conv4(x_4_cat_pred)

        return x_4_cat_pred, x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DeepLab(nn.Module):
    def __init__(self, output_stride, class_num, bn_momentum=0.1, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.mobilenet = MobileNetV2(output_stride=16)
        # added code by ykchua1 (loading pre-trained params)
        pretrain_dict = torch.load('./deepLrn/models/mobilenet_VOC.pth')
        model_dict = {}
        state_dict = self.mobilenet.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.mobilenet.load_state_dict(state_dict)
        
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(class_num, bn_momentum)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low_level_features = self.mobilenet(input)

        x = self.encoder(x)
        predict, intermed = self.decoder(x, low_level_features)
        output= F.interpolate(predict, size=input.size()[2:4], mode='bilinear', align_corners=True)
        return output, intermed

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


if __name__ =="__main__":
    model = DeepLab(output_stride=16, class_num=21, freeze_bn=False)
    model.eval()
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, (3, 512, 512))
    # for m in model.named_modules():
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
