import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
# from .util_relu import conv, predict_flow, deconv, crop_like

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'spynet', 'spynet_bn'
]

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)

class SpyNet(nn.Module):
    def __init__(self, batchNorm=True):
        super(SpyNet, self).__init__()

        self.batchNorm = batchNorm

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, tenInput):
                return self.netBasic(tenInput)

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

    def forward(self, input_ten):
        tenFirst = input_ten[:,:3,:,:]
        tenSecond = input_ten[:,3:,:,:]
        
        tenFlow = []
        tenFirst = [tenFirst]
        tenSecond = [tenSecond]

        for intLevel in range(5):
            if tenFirst[0].shape[2] > 32 or tenFirst[0].shape[3] > 32:
                tenFirst.insert(0, torch.nn.functional.avg_pool2d(input=tenFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                tenSecond.insert(0, torch.nn.functional.avg_pool2d(input=tenSecond[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end
        # for tensize in tenFirst:
        #     print ('tensize', tensize.size())

        # print ('tenFirst', len(tenFirst))
        # print ('tneflowinputsize', [ tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0)) ])

        tenFlow = tenFirst[0].new_zeros([ tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0)) ])
        tenFlow_list = []
        # print ('tenFlowsize', tenFlow.size())
        # print ('++++++++++')
        for intLevel in range(len(tenFirst)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')
            # print ('tenUpsampled', tenUpsampled.size())
            # print ('torchcat_size', torch.cat([ tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1).size())
            tenFlow = self.netBasic[intLevel](torch.cat([ tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
            tenFlow_list.insert(0, tenFlow)
            # print ('eachtenFlow', tenFlow.size())
            # print ('===================')
        # end
        # print ('FinaltenFlow', tenFlow.size())
        # return tenFlow
        if self.training:
            return tenFlow_list[0],tenFlow_list[1],tenFlow_list[2],tenFlow_list[3],tenFlow_list[4]
        else:
            return tenFlow_list[0]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def spynet(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = SpyNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def spynet_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
