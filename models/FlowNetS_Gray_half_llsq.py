import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util_lsq import conv, predict_flow, deconv, crop_like, conv_Q, predict_flow_Q, deconv_Q, ACT_Q
from .util_lsq import QuantConvTranspose2d as ConvTrans2d_Q

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'fnsghllsq', 'fnsghllsq_bn'
]


class FlowNetSGrayHalfLLSQ(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True, bitW=32, bitA=32, cut_ratio=2):
        super(FlowNetSGrayHalfLLSQ,self).__init__()

        ratio = cut_ratio
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
        self.conv1   = conv_Q(self.batchNorm,   2,   C01_OUT, bitW=bitW, bitA=bitA) # 7x7 origin
        self.conv1_1 = conv_Q(self.batchNorm,  C01_OUT,   C11_OUT, bitW=bitW, bitA=bitA)
        self.conv1_2 = conv_Q(self.batchNorm,  C11_OUT,   C12_OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv2   = conv_Q(self.batchNorm,  C12_OUT,  C2__OUT, bitW=bitW, bitA=bitA) # 5x5 origin
        self.conv2_1 = conv_Q(self.batchNorm, C2__OUT,  C21_OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv3   = conv_Q(self.batchNorm, C21_OUT,  C3__OUT, bitW=bitW, bitA=bitA) # 5x5 origin
        self.conv3_0 = conv_Q(self.batchNorm, C3__OUT,  C30_OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv3_1 = conv_Q(self.batchNorm, C30_OUT,  C31_OUT, bitW=bitW, bitA=bitA)
        self.conv4   = conv_Q(self.batchNorm, C31_OUT,  C4__OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv4_1 = conv_Q(self.batchNorm, C4__OUT,  C41_OUT, bitW=bitW, bitA=bitA)
        self.conv5   = conv_Q(self.batchNorm, C41_OUT,  C5__OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv5_1 = conv_Q(self.batchNorm, C5__OUT,  C51_OUT, bitW=bitW, bitA=bitA)
        self.conv6   = conv_Q(self.batchNorm, C51_OUT, C6__OUT, stride=2, bitW=bitW, bitA=bitA)
        self.conv6_1 = conv_Q(self.batchNorm,C6__OUT, C61_OUT, bitW=bitW, bitA=bitA)

        self.deconv5 = deconv_Q(C61_OUT,DC5_OUT, bitW=bitW, bitA=bitA)
        self.deconv4 = deconv_Q(C51_OUT+DC5_OUT+2,DC4_OUT, bitW=bitW, bitA=bitA)
        self.deconv3 = deconv_Q(C41_OUT+DC4_OUT+2,DC3_OUT, bitW=bitW, bitA=bitA)
        self.deconv2 = deconv_Q(C31_OUT+DC3_OUT+2,DC2_OUT, bitW=bitW, bitA=bitA)

        self.predict_flow6 = predict_flow_Q(C61_OUT, bitW=bitW)
        self.predict_flow5 = predict_flow_Q(C51_OUT+DC5_OUT+2, bitW=bitW)
        self.predict_flow4 = predict_flow_Q(C41_OUT+DC4_OUT+2, bitW=bitW)
        self.predict_flow3 = predict_flow_Q(C31_OUT+DC3_OUT+2, bitW=bitW)
        self.predict_flow2 = predict_flow_Q(C21_OUT+DC2_OUT+2, bitW=bitW)

        k_up = 4
        self.upsampled_flow6_to_5 = ConvTrans2d_Q(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = ConvTrans2d_Q(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = ConvTrans2d_Q(2, 2, k_up, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = ConvTrans2d_Q(2, 2, k_up, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        # x = torch.cat(input_ten,1).to(device)
        # p = input_ten[:,0:1,:,:]
        # p.repeat(1,3,1,1)
        # q = input_ten[:,1,:,:]
        # q.repeat(1,3,1,1)
        # x = torch.cat([p,q], 1)
        # x = torch.cat([x[:,:1,:,:].repeat(1,3,1,1),x[:,1:,:,:].repeat(1,3,1,1)],1)

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

def fnsghllsq(data=None, bitW=32, bitA=32, cut_ratio=2):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetSGrayHalfLLSQ(batchNorm=False, bitW=bitW, bitA=bitA, cut_ratio=cut_ratio)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


def fnsghllsq_bn(data=None, bitW=32, bitA=32, cut_ratio=2):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetSGrayHalfLLSQ(batchNorm=False, bitW=bitW, bitA=bitA, cut_ratio=cut_ratio)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model
