import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util_relu import conv, predict_flow, deconv, crop_like

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'knet', 'knet_bn'
]


class KNet(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(KNet,self).__init__()

        ratio = 8
        C11_OUT = 64//ratio
        C12_OUT = 64//ratio
        C21_OUT = 64//ratio
        C22_OUT = 128//ratio
        C31_OUT = 128//ratio
        C32_OUT = 256//ratio
        C41_OUT = 256//ratio
        C42_OUT = 256//ratio

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5

        UF2_OUT = 2
        UF2_OUT = 2
        UF2_OUT = 2
        UF2_OUT = 2

        self.batchNorm = batchNorm
        self.conv1_1 = conv(self.batchNorm,       3, C11_OUT)
        self.conv1_2 = conv(self.batchNorm, C11_OUT, C12_OUT)
        
        self.conv2_1 = conv(self.batchNorm, C12_OUT, C21_OUT)
        self.conv2_2 = conv(self.batchNorm, C21_OUT, C22_OUT)
        
        self.conv3_1 = conv(self.batchNorm, C22_OUT, C31_OUT)
        self.conv3_2 = conv(self.batchNorm, C31_OUT, C32_OUT)

        self.conv4_1 = conv(self.batchNorm, C32_OUT, C41_OUT)
        self.conv4_2 = conv(self.batchNorm, C41_OUT, C42_OUT)
        
        # self.conv5_1 = conv(self.batchNorm, C42_OUT, C51_OUT)
        # self.conv5_2 = conv(self.batchNorm, C51_OUT, C52_OUT)


        self.enco1_1 = conv(self.batchNorm, C12_OUT*2+6+2, EC11_OUT) # c12,c22,im1,im2,flow2_up
        self.enco1_2 = conv(self.batchNorm, EC11_OUT, EC12_OUT)
        self.enco1_3 = conv(self.batchNorm, EC12_OUT, EC13_OUT)
        self.enco1_4 = conv(self.batchNorm, EC13_OUT, EC14_OUT)
        self.enco1_5 = conv(self.batchNorm, EC14_OUT, EC15_OUT)

        self.enco2_1 = conv(self.batchNorm, C22_OUT*2+C12_OUT*2+2 EC21_OUT) # c12,c22,p12,p22,flow3_up
        self.enco2_2 = conv(self.batchNorm, EC21_OUT, EC22_OUT)
        self.enco2_3 = conv(self.batchNorm, EC22_OUT, EC23_OUT)        
        self.enco2_4 = conv(self.batchNorm, EC23_OUT, EC24_OUT)
        self.enco2_5 = conv(self.batchNorm, EC24_OUT, EC25_OUT)

        self.enco3_1 = conv(self.batchNorm, C32_OUT*2+C32_OUT*2+2 EC31_OUT) # c13,c23,p13,p23,flow4_up
        self.enco3_2 = conv(self.batchNorm, EC31_OUT, EC32_OUT)
        self.enco3_3 = conv(self.batchNorm, EC32_OUT, EC33_OUT)
        self.enco3_4 = conv(self.batchNorm, EC33_OUT, EC34_OUT)
        self.enco3_5 = conv(self.batchNorm, EC34_OUT, EC35_OUT)

        self.enco4_1 = conv(self.batchNorm, C42_OUT*2+C42_OUT*2+2 EC41_OUT) # c14,c24,p14,p24,flow5_up
        self.enco4_2 = conv(self.batchNorm, EC41_OUT, EC42_OUT)
        self.enco4_3 = conv(self.batchNorm, EC42_OUT, EC43_OUT)
        self.enco4_4 = conv(self.batchNorm, EC43_OUT, EC44_OUT)
        self.enco4_5 = conv(self.batchNorm, EC44_OUT, EC45_OUT)

        self.enco5_1 = conv(self.batchNorm, C42_OUT*2+2 EC31_OUT) # p15,p25
        self.enco5_2 = conv(self.batchNorm, EC51_OUT, EC52_OUT)
        self.enco5_3 = conv(self.batchNorm, EC52_OUT, EC53_OUT)
        self.enco5_4 = conv(self.batchNorm, EC53_OUT, EC54_OUT)
        self.enco5_5 = conv(self.batchNorm, EC54_OUT, EC55_OUT)

        # self.deconv2 = deconv(EC22_OUT,DC2_OUT)
        # self.deconv3 = deconv(EC32_OUT,DC3_OUT)
        # self.deconv4 = deconv(EC43_OUT,DC4_OUT)
        # self.deconv5 = deconv(EC53_OUT,DC5_OUT)

        self.upflow2 = nn.ConvTranspose2d(EC25_OUT, UF2_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow3 = nn.ConvTranspose2d(EC35_OUT, UF3_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow4 = nn.ConvTranspose2d(EC45_OUT, UF4_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow5 = nn.ConvTranspose2d(EC55_OUT, UF5_OUT, kernel_size=4, stride=2, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        c11 = self.conv1_2(self.conv1_1(im1))
        c21 = self.conv1_2(self.conv1_1(im2))
        
        p12 = nn.functional.avg_pool2d(input=c11, kernel_size=2, stride=2, count_include_pad=False)
        p22 = nn.functional.avg_pool2d(input=c12, kernel_size=2, stride=2, count_include_pad=False)        
        c12 = self.conv2_2(self.conv2_1(p12))
        c22 = self.conv2_2(self.conv2_1(p22))
        
        p13 = nn.functional.avg_pool2d(input=c12, kernel_size=2, stride=2, count_include_pad=False)
        p23 = nn.functional.avg_pool2d(input=c22, kernel_size=2, stride=2, count_include_pad=False)
        c13 = self.conv3_2(self.conv3_1(p13))
        c23 = self.conv3_2(self.conv3_1(p23))
        
        p14 = nn.functional.avg_pool2d(input=c13, kernel_size=2, stride=2, count_include_pad=False)
        p24 = nn.functional.avg_pool2d(input=c13, kernel_size=2, stride=2, count_include_pad=False)
        c14 = self.conv4_2(self.conv4_1(p14))
        c24 = self.conv4_2(self.conv4_1(p24))
        
        p15 = nn.functional.avg_pool2d(input=c14, kernel_size=2, stride=2, count_include_pad=False)
        p25 = nn.functional.avg_pool2d(input=c24, kernel_size=2, stride=2, count_include_pad=False)
        # c15 = self.conv1_2(self.conv1_1(p15))
        # c25 = self.conv1_2(self.conv1_1(p25))

        enco51 = self.enco5_1(torch.cat((p15,p25),1))
        enco52 = self.enco5_2(enco51)
        enco53 = self.enco5_3(enco52)
        enco54 = self.enco5_4(enco53)
        flow5 = self.enco5_5(enco54)
        flow5_up = self.upflow5(flow5)

        enco41 = self.enco4_1(torch.cat((c14,c24,p14,p24,flow5_up),1))
        enco42 = self.enco4_2(enco41)
        enco43 = self.enco4_3(enco42)
        enco44 = self.enco4_4(enco43)
        flow4 = self.enco4_5(enco44)
        flow4_up = self.upflow4(flow4)
        
        enco31 = self.enco3_1(torch.cat((c13,c23,p13,p23,flow4_up),1))
        enco32 = self.enco3_2(enco31)
        enco33 = self.enco3_3(enco32)
        enco34 = self.enco3_4(enco33)
        flow3 = self.enco3_5(enco34)
        flow3_up = self.upflow3(flow3)

        enco21 = self.enco2_1(torch.cat((c12,c22,p12,p22,flow3_up),1))
        enco22 = self.enco2_2(enco21)
        enco23 = self.enco2_3(enco22)
        enco24 = self.enco2_4(enco23)
        flow2 = self.enco2_5(enco24)
        flow2_up = self.upflow2(flow2)

        enco11 = self.enco1_1(torch.cat((c11,c21,im1,im2,flow2_up),1))
        enco12 = self.enco1_2(enco21)
        enco13 = self.enco1_4(enco22)
        enco14 = self.enco1_3(enco23)
        flow1 = self.enco1_5(enco24)

        if self.training:
            return flow1,flow2,flow3,flow4,flow5
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetsgrayhalf(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = KNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def flownetsgrayhalf_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = KNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
