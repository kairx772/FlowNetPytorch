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

        C11_OUT = 32
        C12_OUT = 32
        C21_OUT = 32
        C22_OUT = 32
        C31_OUT = 32
        C32_OUT = 32
        C41_OUT = 32
        C42_OUT = 32
        C51_OUT = 32
        C52_OUT = 32

        ecchnum1 = 64
        ecchnum2 = 64
        ecchnum3 = 128
        ecchnum4 = 128
        ecchnum5 = 256
        ecchnum6 = 256
        ecchnum7 = 128
        ecchnum8 = 64
        ecchnum9 = 2

        EC11_OUT = ecchnum1
        EC12_OUT = ecchnum2
        EC13_OUT = ecchnum3
        EC14_OUT = ecchnum4
        EC15_OUT = ecchnum5
        EC16_OUT = ecchnum6
        EC17_OUT = ecchnum7
        EC18_OUT = ecchnum8
        EC19_OUT = ecchnum9

        EC21_OUT = ecchnum1
        EC22_OUT = ecchnum2
        EC23_OUT = ecchnum3
        EC24_OUT = ecchnum4
        EC25_OUT = ecchnum5
        EC26_OUT = ecchnum6
        EC27_OUT = ecchnum7
        EC28_OUT = ecchnum8
        EC29_OUT = ecchnum9

        EC31_OUT = ecchnum1
        EC32_OUT = ecchnum2
        EC33_OUT = ecchnum3
        EC34_OUT = ecchnum4
        EC35_OUT = ecchnum5
        EC36_OUT = ecchnum5
        EC37_OUT = ecchnum5
        EC38_OUT = ecchnum5
        EC39_OUT = ecchnum5

        EC41_OUT = ecchnum1
        EC42_OUT = ecchnum2
        EC43_OUT = ecchnum3
        EC44_OUT = ecchnum4
        EC45_OUT = ecchnum5
        EC46_OUT = ecchnum6
        EC47_OUT = ecchnum7
        EC48_OUT = ecchnum8
        EC49_OUT = ecchnum9

        EC51_OUT = ecchnum1
        EC52_OUT = ecchnum2
        EC53_OUT = ecchnum3
        EC54_OUT = ecchnum4
        EC55_OUT = ecchnum5
        EC56_OUT = ecchnum6
        EC57_OUT = ecchnum7
        EC58_OUT = ecchnum8
        EC59_OUT = ecchnum9

        UF2_OUT = 2
        UF3_OUT = 2
        UF4_OUT = 2
        UF5_OUT = 2

        self.batchNorm = batchNorm
        self.conv1_1 = conv(self.batchNorm,       3, C11_OUT)
        self.conv1_2 = conv(self.batchNorm, C11_OUT, C12_OUT)
        
        self.conv2_1 = conv(self.batchNorm, C12_OUT, C21_OUT)
        self.conv2_2 = conv(self.batchNorm, C21_OUT, C22_OUT)
        
        self.conv3_1 = conv(self.batchNorm, C22_OUT, C31_OUT)
        self.conv3_2 = conv(self.batchNorm, C31_OUT, C32_OUT)

        self.conv4_1 = conv(self.batchNorm, C32_OUT, C41_OUT)
        self.conv4_2 = conv(self.batchNorm, C41_OUT, C42_OUT)
        
        self.conv5_1 = conv(self.batchNorm, C42_OUT, C51_OUT)
        self.conv5_2 = conv(self.batchNorm, C51_OUT, C52_OUT)


        self.enco1_1 = conv(self.batchNorm, C12_OUT*2+6+2, EC11_OUT) # c12,c22,im1,im2,flow2_up
        self.enco1_2 = conv(self.batchNorm, EC11_OUT, EC12_OUT)
        self.enco1_3 = conv(self.batchNorm, EC12_OUT, EC13_OUT)
        self.enco1_4 = conv(self.batchNorm, EC13_OUT, EC14_OUT)
        self.enco1_5 = conv(self.batchNorm, EC14_OUT, EC15_OUT)
        self.enco1_6 = conv(self.batchNorm, EC15_OUT, EC16_OUT)
        self.enco1_7 = conv(self.batchNorm, EC16_OUT, EC17_OUT)
        self.enco1_8 = conv(self.batchNorm, EC17_OUT, EC18_OUT)
        self.enco1_9 = conv(self.batchNorm, EC18_OUT, EC19_OUT)

        self.enco2_1 = conv(self.batchNorm, C22_OUT*2+C12_OUT*2+2, EC21_OUT) # c12,c22,p12,p22,flow3_up
        self.enco2_2 = conv(self.batchNorm, EC21_OUT, EC22_OUT)
        self.enco2_3 = conv(self.batchNorm, EC22_OUT, EC23_OUT)        
        self.enco2_4 = conv(self.batchNorm, EC23_OUT, EC24_OUT)
        self.enco2_5 = conv(self.batchNorm, EC24_OUT, EC25_OUT)
        self.enco2_6 = conv(self.batchNorm, EC25_OUT, EC26_OUT)
        self.enco2_7 = conv(self.batchNorm, EC26_OUT, EC27_OUT)
        self.enco2_8 = conv(self.batchNorm, EC27_OUT, EC28_OUT)
        self.enco2_9 = conv(self.batchNorm, EC28_OUT, EC29_OUT)

        self.enco3_1 = conv(self.batchNorm, C32_OUT*2+C22_OUT*2+2, EC31_OUT) # c13,c23,p13,p23,flow4_up
        self.enco3_2 = conv(self.batchNorm, EC31_OUT, EC32_OUT)
        self.enco3_3 = conv(self.batchNorm, EC32_OUT, EC33_OUT)
        self.enco3_4 = conv(self.batchNorm, EC33_OUT, EC34_OUT)
        self.enco3_5 = conv(self.batchNorm, EC34_OUT, EC35_OUT)
        self.enco3_6 = conv(self.batchNorm, EC35_OUT, EC36_OUT)
        self.enco3_7 = conv(self.batchNorm, EC36_OUT, EC37_OUT)
        self.enco3_8 = conv(self.batchNorm, EC37_OUT, EC38_OUT)
        self.enco3_9 = conv(self.batchNorm, EC38_OUT, EC39_OUT)

        self.enco4_1 = conv(self.batchNorm, C42_OUT*2+C32_OUT*2+2, EC41_OUT) # c14,c24,p14,p24,flow5_up
        self.enco4_2 = conv(self.batchNorm, EC41_OUT, EC42_OUT)
        self.enco4_3 = conv(self.batchNorm, EC42_OUT, EC43_OUT)
        self.enco4_4 = conv(self.batchNorm, EC43_OUT, EC44_OUT)
        self.enco4_5 = conv(self.batchNorm, EC44_OUT, EC45_OUT)
        self.enco4_6 = conv(self.batchNorm, EC45_OUT, EC46_OUT)
        self.enco4_7 = conv(self.batchNorm, EC46_OUT, EC47_OUT)
        self.enco4_8 = conv(self.batchNorm, EC47_OUT, EC48_OUT)
        self.enco4_9 = conv(self.batchNorm, EC48_OUT, EC49_OUT)

        self.enco5_1 = conv(self.batchNorm, C52_OUT*2+C42_OUT*2, EC51_OUT) # p15,p25
        self.enco5_2 = conv(self.batchNorm, EC51_OUT, EC52_OUT)
        self.enco5_3 = conv(self.batchNorm, EC52_OUT, EC53_OUT)
        self.enco5_4 = conv(self.batchNorm, EC53_OUT, EC54_OUT)
        self.enco5_5 = conv(self.batchNorm, EC54_OUT, EC55_OUT)
        self.enco5_6 = conv(self.batchNorm, EC55_OUT, EC56_OUT)
        self.enco5_7 = conv(self.batchNorm, EC56_OUT, EC57_OUT)
        self.enco5_8 = conv(self.batchNorm, EC57_OUT, EC58_OUT)
        self.enco5_9 = conv(self.batchNorm, EC58_OUT, EC59_OUT)

        # self.deconv2 = deconv(EC22_OUT,DC2_OUT)
        # self.deconv3 = deconv(EC32_OUT,DC3_OUT)
        # self.deconv4 = deconv(EC43_OUT,DC4_OUT)
        # self.deconv5 = deconv(EC53_OUT,DC5_OUT)

        self.upflow2 = nn.ConvTranspose2d(EC29_OUT, UF2_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow3 = nn.ConvTranspose2d(EC39_OUT, UF3_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow4 = nn.ConvTranspose2d(EC49_OUT, UF4_OUT, kernel_size=4, stride=2, padding=1, bias=True)
        self.upflow5 = nn.ConvTranspose2d(EC59_OUT, UF5_OUT, kernel_size=4, stride=2, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        im1 = x[:,:3,:,:]
        im2 = x[:,3:,:,:]

        c11 = self.conv1_2(self.conv1_1(im1))
        c12 = self.conv1_2(self.conv1_1(im2))
        
        p21 = nn.functional.avg_pool2d(input=c11, kernel_size=2, stride=2, count_include_pad=False)
        p22 = nn.functional.avg_pool2d(input=c12, kernel_size=2, stride=2, count_include_pad=False)        
        c21 = self.conv2_2(self.conv2_1(p21))
        c22 = self.conv2_2(self.conv2_1(p22))
        
        p31 = nn.functional.avg_pool2d(input=c21, kernel_size=2, stride=2, count_include_pad=False)
        p32 = nn.functional.avg_pool2d(input=c22, kernel_size=2, stride=2, count_include_pad=False)
        c31 = self.conv3_2(self.conv3_1(p31))
        c32 = self.conv3_2(self.conv3_1(p32))
        
        p41 = nn.functional.avg_pool2d(input=c31, kernel_size=2, stride=2, count_include_pad=False)
        p42 = nn.functional.avg_pool2d(input=c32, kernel_size=2, stride=2, count_include_pad=False)
        c41 = self.conv4_2(self.conv4_1(p41))
        c42 = self.conv4_2(self.conv4_1(p42))
        
        p51 = nn.functional.avg_pool2d(input=c41, kernel_size=2, stride=2, count_include_pad=False)
        p52 = nn.functional.avg_pool2d(input=c42, kernel_size=2, stride=2, count_include_pad=False)
        c51 = self.conv5_2(self.conv5_1(p51))
        c52 = self.conv5_2(self.conv5_1(p52))

        # c15 = self.conv1_2(self.conv1_1(p15))
        # c25 = self.conv1_2(self.conv1_1(p25))

        enco51 = self.enco5_1(torch.cat((c51,c52,p51,p52),1))
        enco52 = self.enco5_2(enco51)
        enco53 = self.enco5_3(enco52)
        enco54 = self.enco5_4(enco53)
        enco55 = self.enco5_5(enco54)
        enco56 = self.enco5_6(enco55)
        enco57 = self.enco5_7(enco56)
        enco58 = self.enco5_8(enco57)
        flow5  = self.enco5_9(enco58)
        flow5_up = self.upflow5(flow5)

        enco41 = self.enco4_1(torch.cat((c41,c42,p41,p42,flow5_up),1))
        enco42 = self.enco4_2(enco41)
        enco43 = self.enco4_3(enco42)
        enco44 = self.enco4_4(enco43)
        enco45 = self.enco4_5(enco44)
        enco46 = self.enco4_6(enco45)
        enco47 = self.enco4_7(enco46)
        enco48 = self.enco4_8(enco47)
        flow4 = self.enco4_9(enco48)
        flow4_up = self.upflow4(flow4)
        
        enco31 = self.enco3_1(torch.cat((c31,c32,p31,p32,flow4_up),1))
        enco32 = self.enco3_2(enco31)
        enco33 = self.enco3_3(enco32)
        enco34 = self.enco3_4(enco33)
        enco35 = self.enco3_5(enco34)
        enco36 = self.enco3_6(enco35)
        enco37 = self.enco3_7(enco36)
        enco38 = self.enco3_8(enco37)
        flow3 = self.enco3_9(enco38)
        flow3_up = self.upflow3(flow3)

        enco21 = self.enco2_1(torch.cat((c21,c22,p21,p22,flow3_up),1))# ; print ('enco21 size:', enco21.size())
        enco22 = self.enco2_2(enco21)#; print ('enco22 size:', enco22.size())
        enco23 = self.enco2_3(enco22)#; print ('enco23 size:', enco23.size())
        enco24 = self.enco2_4(enco23)#; print ('enco24 size:', enco24.size())
        enco25 = self.enco2_5(enco24)
        enco26 = self.enco2_6(enco25)
        enco27 = self.enco2_7(enco26)
        enco28 = self.enco2_8(enco27)
        flow2 = self.enco2_9(enco28)#; print ('flow2 size:', flow2.size())
        flow2_up = self.upflow2(flow2)

        enco11 = self.enco1_1(torch.cat((c11,c12,im1,im2,flow2_up),1))
        #print ('enco11 size:', enco11.size())
        enco12 = self.enco1_2(enco11)
        #rint ('enco12 size:', enco12.size())
        enco13 = self.enco1_3(enco12)
        enco14 = self.enco1_4(enco13)
        enco15 = self.enco1_5(enco14)
        enco16 = self.enco1_6(enco15)
        enco17 = self.enco1_7(enco16)
        enco18 = self.enco1_8(enco17)
        flow1 = self.enco1_9(enco18)

        if self.training:
            return flow1,flow2,flow3,flow4,flow5
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def knet(data=None, args=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = KNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def knet_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = KNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
