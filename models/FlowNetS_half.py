import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util_relu import conv, predict_flow, deconv, crop_like

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'flownetshalf', 'flownetshalf_bn'
]


class FlowNetShalf(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(FlowNetSGray,self).__init__()

        ratio = 2
        C01_OUT = 64//ratio
        C11_OUT = 64//ratio
        C12_OUT = 64//ratio
        C2__OUT = 128//ratio
        C21_OUT = 128//ratio
        C3__OUT = 256//ratio
        C30_OUT = 256//ratio
        C31_OUT = 256//ratio
        C4__OUT = 512//ratio
        C41_OUT = 512//ratio
        C5__OUT = 512//ratio
        C51_OUT = 512//ratio
        C6__OUT = 1024//ratio
        C61_OUT = 1024//ratio

        DC5_OUT = 512//ratio
        DC4_OUT = 256//ratio
        DC3_OUT = 128//ratio
        DC2_OUT = 64//ratio

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   C01_OUT) # 7x7 origin
        self.conv1_1 = conv(self.batchNorm,  C01_OUT,   C11_OUT)
        self.conv1_2 = conv(self.batchNorm,  C11_OUT,   C12_OUT, stride=2)
        self.conv2   = conv(self.batchNorm,  C12_OUT,  C2__OUT) # 5x5 origin
        self.conv2_1 = conv(self.batchNorm, C2__OUT,  C21_OUT, stride=2)
        self.conv3   = conv(self.batchNorm, C21_OUT,  C3__OUT) # 5x5 origin
        self.conv3_0 = conv(self.batchNorm, C3__OUT,  C30_OUT, stride=2)
        self.conv3_1 = conv(self.batchNorm, C30_OUT,  C31_OUT)
        self.conv4   = conv(self.batchNorm, C31_OUT,  C4__OUT, stride=2)
        self.conv4_1 = conv(self.batchNorm, C4__OUT,  C41_OUT)
        self.conv5   = conv(self.batchNorm, C41_OUT,  C5__OUT, stride=2)
        self.conv5_1 = conv(self.batchNorm, C5__OUT,  C51_OUT)
        self.conv6   = conv(self.batchNorm, C51_OUT, C6__OUT, stride=2)
        self.conv6_1 = conv(self.batchNorm,C6__OUT, C61_OUT)

        self.deconv5 = deconv(C61_OUT,DC5_OUT)
        self.deconv4 = deconv(C51_OUT+DC5_OUT+2,DC4_OUT)
        self.deconv3 = deconv(C41_OUT+DC4_OUT+2,DC3_OUT)
        self.deconv2 = deconv(C31_OUT+DC3_OUT+2,DC2_OUT)

        self.predict_flow6 = predict_flow(C61_OUT)
        self.predict_flow5 = predict_flow(C51_OUT+DC5_OUT+2)
        self.predict_flow4 = predict_flow(C41_OUT+DC4_OUT+2)
        self.predict_flow3 = predict_flow(C31_OUT+DC3_OUT+2)
        self.predict_flow2 = predict_flow(C21_OUT+DC2_OUT+2)

        k_up = 4
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, k_up, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        out_conv2 = self.conv2_1(self.conv2(self.conv1_2(self.conv1_1(self.conv1(x)))))
        out_conv3 = self.conv3_1(self.conv3_0(self.conv3(out_conv2)))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetshalf(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetShalf(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def flownetshalf_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetShalf(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
