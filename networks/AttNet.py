from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttNet(nn.Module):

    def __init__(self, num_input_ch):
        super(AttNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.num_input_ch = num_input_ch
        self.conv1 = nn.Conv2d(self.num_input_ch, 64, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 16, 1, bias=True)
        self.conv3 = nn.Conv2d(16, 1, 1, bias=True)
        

    def forward(self, warp_feat, conv_feat):
        concat_feat = torch.cat([warp_feat, conv_feat], dim=0)
        weights = F.relu(self.conv1(concat_feat))
        weights = F.relu(self.conv2(weights))
        weights = F.softmax(self.conv3(weights), dim=0)
        weights = torch.split(weights, 2, dim=0)
        weight1 = torch.tile(weights[0], (1, self.num_input_ch, 1, 1))
        weight2 = torch.tile(weights[0], (1, self.num_input_ch, 1, 1))
        out_feat = weight1 * warp_feat + weight2 * conv_feat
        return out_feat


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AttNet(3).to(device)
# summary(model, [(3, 800, 800), (3, 800, 800)])
