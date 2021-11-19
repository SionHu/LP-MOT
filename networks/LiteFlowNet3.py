
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn

from correlation import Correlation
from torchsummary import summary

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

##########################################################


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        self.netOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    # end

    def forward(self, tenInput):
        tenOne = self.netOne(tenInput)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
    # end
# end

class Matching(nn.Module):
    def __init__(self, intLevel):
        super(Matching, self).__init__()

        self.fltBackwarp = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]


        self.crossCorr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)


        if intLevel == 4:
            self.autoCorr = Correlation(pad_size=6, kernel_size=1, max_displacement=6, stride1=1, stride2=2)
        elif intLevel == 3:
            self.autoCorr = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=2)

        if intLevel > 4:
            self.confFeat = None
            self.corrFeat = None

        if intLevel <= 4:
            self.confFeat = nn.Sequential(
                nn.Conv2d(in_channels=[0, 0, 0, 1 + 81, 1 + 49][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.dispNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
                )
            self.confNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid()
                )

            self.corrFeat = nn.Sequential(
                nn.Conv2d(in_channels=[0, 0, 0, 64 + 81 + 1, 96 + 81 + 1][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
            self.corrScalar = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
            )
            self.corrOffset = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
            )

        # end

        if intLevel == 6:
            self.netUpflow = None

        elif intLevel != 6:
            self.netUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

        if intLevel == 4 or intLevel == 3:
            self.netUpconf = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False, groups=1)
        # end

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=81, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
        )
    # end

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow, tenConf):
        if self.confFeat:
            tenConf = self.netUpconf(tenConf)
            tenCorrelation = nn.functional.leaky_relu(input=self.autoCorr(tenFeaturesFirst, tenFeaturesFirst), negative_slope=0.1, inplace=False)
            confFeat = self.confFeat(torch.cat([tenCorrelation, tenConf], 1))
            tenConf = self.confNet(confFeat)
            tenDisp = self.dispNet(confFeat)

        if tenFlow is not None:
            tenFlow = self.netUpflow(tenFlow)
        # end
        if self.corrFeat:
            tenFlow = backwarp(tenInput=tenFlow, tenFlow=tenDisp)

        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackwarp)
        # end

        tenCorrelation = nn.functional.leaky_relu(input=self.crossCorr(tenFeaturesFirst, tenFeaturesSecond), negative_slope=0.1, inplace=False)


        if self.corrFeat:
            corrfeat = self.corrFeat(torch.cat([tenFeaturesFirst, tenCorrelation, tenConf], 1))
            corrscalar = self.corrScalar(corrfeat)
            corroffset = self.corrOffset(corrfeat)
            tenCorrelation = corrscalar * tenCorrelation + corroffset

        return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation), tenConf
    # end
# end

class Subpixel(nn.Module):
    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 0, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
        )
    # end

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackward)
        # end

        return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesFirst, tenFeaturesSecond, tenFlow ], 1))
    # end
# end

class Regularization(nn.Module):
    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

        if intLevel > 4:
            self.netFeat = nn.Sequential()

        elif intLevel <= 4:
            self.netFeat = nn.Sequential(
                nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        # end

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )



        if intLevel >= 5:
            self.netDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
            )

        elif intLevel < 5:
            self.netDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 0, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 0, 2, 2, 1, 1 ][intLevel], 0)),
                nn.Conv2d(in_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 0, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 0, 2, 2, 1, 1 ][intLevel]))
            )

        if intLevel == 5 or intLevel == 4:
            self.confNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=[0, 0, 0, 0, 5, 3][intLevel], stride=1, padding=[0, 0, 0, 0, 2, 1][intLevel]),
                nn.Sigmoid()
            )
        else:
            self.confNet = None
        # end

        self.netScaleX = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScaleY = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
    # eny

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenDifference = (tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackward)).pow(2.0).sum(1, True).sqrt().detach()

        tenFeaturesFirst = self.netFeat(tenFeaturesFirst)

        mainfeat = self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), tenFeaturesFirst ], 1))
        tenDist = self.netDist(mainfeat)

        tenConf = None
        if self.confNet:
            tenConf = self.confNet(mainfeat)

        tenDist = tenDist.pow(2.0).neg()
        tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

        tenDivisor = tenDist.sum(1, True).reciprocal()

        tenScaleX = self.netScaleX(tenDist * nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
        tenScaleY = self.netScaleY(tenDist * nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

        return torch.cat([ tenScaleX, tenScaleY ], 1), tenConf
    # end
# end

class LiteFlowNet3(nn.Module):
    def __init__(self):
        super(LiteFlowNet3, self).__init__()

        self.netFeatures = Features()
        self.netMatching = nn.ModuleList([ Matching(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netSubpixel = nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
        self.netRegularization = nn.ModuleList([ Regularization(intLevel) for intLevel in [ 3, 4, 5, 6 ] ])
    # end

    def forward(self, tenFirst, tenSecond):

        tenFirst = tenFirst - torch.mean(tenFirst, (2, 3), keepdim=True)
        tenSecond = tenSecond - torch.mean(tenSecond, (2, 3), keepdim=True)

        # tenFirst = torch.from_numpy(np.load('../caffe_output/img0_nomean_resize.npy')).cuda()
        # tenSecond = torch.from_numpy(np.load('../caffe_output/img1_nomean_resize.npy')).cuda()

        tenFeaturesFirst = self.netFeatures(tenFirst)
        tenFeaturesSecond = self.netFeatures(tenSecond)

        # for idx in range(len(tenFeaturesFirst)):
        #     print('diff_F0_L{}'.format(idx+1), torch.max(torch.abs(tenFeaturesFirst[idx]-load_caffe_output('F0_L{}'.format(idx+1)))))
        #     print('diff_F1_L{}'.format(idx+1), torch.max(torch.abs(tenFeaturesSecond[idx]-load_caffe_output('F1_L{}'.format(idx+1)))))

        tenFirst = [ tenFirst ]
        tenSecond = [ tenSecond ]

        for intLevel in [ 2, 3, 4, 5 ]:
            tenFirst.append(nn.functional.interpolate(input=tenFirst[-1], size=(tenFeaturesFirst[intLevel].shape[2], tenFeaturesFirst[intLevel].shape[3]), mode='bilinear', align_corners=False))
            tenSecond.append(nn.functional.interpolate(input=tenSecond[-1], size=(tenFeaturesSecond[intLevel].shape[2], tenFeaturesSecond[intLevel].shape[3]), mode='bilinear', align_corners=False))
        # end

        tenFlow = None
        tenConf = None

        for intLevel in [ -1, -2, -3, -4 ]:
            tenFlow, tenConf = self.netMatching[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow, tenConf)
            tenFlow = self.netSubpixel[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
            tenFlow, tenConf = self.netRegularization[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
        # end

        return tenFlow * 20.0
    # end
# end

##########################################################

arguments_strModel = '/home/hu440/LP-MOT/pytorch-liteflownet3/network-sintel.pytorch'
arguments_strFirst = '/home/hu440/LP-MOT/pytorch-liteflownet3/images/first.png'
arguments_strSecond = '/home/hu440/LP-MOT/pytorch-liteflownet3/images/second.png'
arguments_strOut = '/home/hu440/LP-MOT/pytorch-liteflownet3/out.flo'

def estimate(netNetwork, tenFirst, tenSecond):
    import math

    # end
    assert(tenFirst.shape[1] == tenSecond.shape[1])
    assert(tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedFirst = nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0)
# end


def read_image(im):
  import PIL.Image
  if not isinstance(im, PIL.Image.Image):
    im = PIL.Image.open(im)
  norm = np.array(im)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
  return torch.FloatTensor(np.ascontiguousarray(norm))


# write_flo in OF
def save_flow(flow, arguments_strOut):
  from utils import OF

  # objOutput = open(arguments_strOut, 'wb')
  # np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
  # np.array([ flow.shape[2], flow.shape[1] ], np.int32).tofile(objOutput)
  # np.array(flow, np.float32).tofile(objOutput)
  # objOutput.close()
  OF.write_flo(flow, arguments_strOut)

  import PIL.Image
  PIL.Image.fromarray(OF.flow_to_image(flow)).save(arguments_strOut + '.png')

#
if __name__ == '__main__':
    im1 = read_image(arguments_strFirst)
    im2 = read_image(arguments_strSecond)
    netNetwork = LiteFlowNet3().cuda().eval()
    netNetwork.load_state_dict(torch.load(arguments_strModel))
    # save_flow(estimate(netNetwork, im1, im2), arguments_strOut)

    print(f' - size: {im1.shape, im2.shape}')
    summary(netNetwork, [(3, 436, 1024), (3, 436, 1024)])
# end
