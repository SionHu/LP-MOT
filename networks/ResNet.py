# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchsummary import summary


class ResNet(nn.Module):
    """Pytorch module for a resnet
    """
    def __init__(self, num_layers, pretrained):
        super(ResNet, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # self.resnet = resnets[num_layers](pretrained)
        self.resnet = resnets[num_layers]()

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        self.features.append(self.resnet.relu(x))
        self.features.append(self.resnet.layer1(self.resnet.maxpool(self.features[-1])))
        self.features.append(self.resnet.layer2(self.features[-1]))
        self.features.append(self.resnet.layer3(self.features[-1]))
        self.features.append(self.resnet.layer4(self.features[-1]))
        
        return self.features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(18, None).to(device)
    print(model)
    summary(model, (3, 270, 480))
