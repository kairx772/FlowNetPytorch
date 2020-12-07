import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util_lsq import conv, predict_flow, deconv, crop_like, conv_Q, predict_flow_Q, deconv_Q, ACT_Q
from .util_lsq import QuantConvTranspose2d as ConvTrans2d_Q

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'flownets33q', 'flownets33q_bn'
]


class FlowNetS33_Q(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True, bitW=32, bitA=32, bias=True):
        super(FlowNetS33_Q,self).__init__()
        
        self.bitW = bitW
        self.bitA = bitA
        print ('bitW = ', bitW)
        print ('bitA = ' , bitA)

        self.batchNorm = batchNorm
        self.conv1   = conv_Q(self.batchNorm,   6,   64, bias=bias, bitW=bitW, bitA=bitA) # 7x7 origin
        self.conv1_1 = conv_Q(self.batchNorm,  64,   64, bias=bias, bitW=bitW, bitA=bitA)
        self.conv1_2 = conv_Q(self.batchNorm,  64,   64, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv2   = conv_Q(self.batchNorm,  64,  128, bias=bias, bitW=bitW, bitA=bitA) # 5x5 origin
        self.conv2_1 = conv_Q(self.batchNorm, 128,  128, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv3   = conv_Q(self.batchNorm, 128,  256, bias=bias, bitW=bitW, bitA=bitA) # 5x5 origin
        self.conv3_0 = conv_Q(self.batchNorm, 256,  256, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv3_1 = conv_Q(self.batchNorm, 256,  256, bias=bias, bitW=bitW, bitA=bitA)
        self.conv4   = conv_Q(self.batchNorm, 256,  512, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv4_1 = conv_Q(self.batchNorm, 512,  512, bias=bias, bitW=bitW, bitA=bitA)
        self.conv5   = conv_Q(self.batchNorm, 512,  512, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv5_1 = conv_Q(self.batchNorm, 512,  512, bias=bias, bitW=bitW, bitA=bitA)
        self.conv6   = conv_Q(self.batchNorm, 512, 1024, bias=bias, bitW=bitW, bitA=bitA, stride=2)
        self.conv6_1 = conv_Q(self.batchNorm,1024, 1024, bias=bias, bitW=bitW, bitA=bitA)

        self.deconv5 = deconv_Q(1024,512, bitW=bitW, bitA=bitA)
        self.deconv4 = deconv_Q(1026,256, bitW=bitW, bitA=bitA)
        self.deconv3 = deconv_Q(770,128, bitW=bitW, bitA=bitA)
        self.deconv2 = deconv_Q(386,64, bitW=bitW, bitA=bitA)

        self.predict_flow6 = predict_flow_Q(1024, bitW=bitW)
        self.predict_flow5 = predict_flow_Q(1026, bitW=bitW)
        self.predict_flow4 = predict_flow_Q(770, bitW=bitW)
        self.predict_flow3 = predict_flow_Q(386, bitW=bitW)
        self.predict_flow2 = predict_flow_Q(194, bitW=bitW)

        self.upsampled_flow6_to_5 = ConvTrans2d_Q(2, 2, 4, 2, 1, bias=False, bit=bitW)
        self.upsampled_flow5_to_4 = ConvTrans2d_Q(2, 2, 4, 2, 1, bias=False, bit=bitW)
        self.upsampled_flow4_to_3 = ConvTrans2d_Q(2, 2, 4, 2, 1, bias=False, bit=bitW)
        self.upsampled_flow3_to_2 = ConvTrans2d_Q(2, 2, 4, 2, 1, bias=False, bit=bitW)

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
    
    def assign_alphabit(self, alphabit):
        for m in self.modules():
            if isinstance(m, ACT_Q):
                m.alpha_bit = alphabit
        return


def flownets33q(data=None, args=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS33_Q(batchNorm=False, bias=args.conv_no_bias, bitW=args.qw, bitA=args.qa)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    if args.alphabit is not None:
        model.assign_alphabit(args.alphabit)
    return model


def flownets33q_bn(data=None, bitW=32, bitA=32):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS33_Q(batchNorm=True, bitW=bitW, bitA=bitA)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
