import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter



try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)


class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        if k == 32:
            return input
        elif k == 1:
            output = torch.sign(input)
        else:
            n = float(2 ** k - 1)
            output = torch.round(input * n ) / n
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None



class fw(nn.Module):
    def __init__(self, bitW):
        super(fw, self).__init__()
        self.bitW = bitW
        self.quantize = quantize().apply
        #self.bn = BNW()
    
    def forward(self, x):
        if self.bitW == 32:
            return x

        elif self.bitW == 1:
            E = torch.mean(torch.abs(x)).detach()
            qx = self.quantize(x / E, self.bitW) * E
            return qx
        else:
            tanh_x = x.tanh()
            max_x = tanh_x.abs().max()
            qx = tanh_x / max_x    
            qx = qx * 0.5 + 0.5
            qx = (2.0 * self.quantize(qx, self.bitW) - 1.0) * max_x
            
            return qx

class Conv2d_Q(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,

                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)

        outputs = F.conv2d(input, q_weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

        return outputs 

class ConvTrans2d_Q(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 bitW=32,

                 dilation=1, 
                 groups=1, 
                 bias=False):
        super(ConvTrans2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input, order=None):
        q_weight = self.fw(self.weight)

        outputs = F.conv_transpose2d(input, q_weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

        return outputs



# class Act_Q(nn.Module):
#     def __init__(self, bitA=32):
#         super(Act_Q, self).__init__()
#         self.bitA = bitA
#         self.quantize = quantize().apply
    
#     def forward(self, x):
#         if self.bitA==32:
#             # max(x, 0.0)
#             qa = torch.nn.functional.relu(x)
#         else:
#             # min(max(x, 0), 1)
#             qa = self.quantize(torch.clamp(x, min=0, max=1), self.bitA)
#         return qa

class ACT_Q(nn.Module):
    def __init__(self,  bit=32 , signed = False, alpha_bit = 32):
        super(ACT_Q, self).__init__()
        #self.inplace    = inplace
        #self.alpha      = Parameter(torch.randn(1), requires_grad=True)
        self.bit        = bit
        self.signed     = signed
        self.pwr_coef   = 2** (bit - 1)
        self.alpha_bit  = alpha_bit
        # self.alpha = Parameter(torch.rand(1))
        if bit < 0:
            self.alpha = None
        else:
            self.alpha = Parameter(torch.rand( 1))
        #self.alpha = Parameter(torch.Tensor(1))    
        self.round_fn = RoundFn_act
        # self.alpha_qfn = quan_fn_alpha()
        if bit < 0:
            self.init_state = 1
        else:
            self.register_buffer('init_state', torch.zeros(1))        #self.init_state = 0

    def forward(self, input):
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(input.detach().abs().max() / (self.pwr_coef + 1))
            self.init_state.fill_(1)
            #self.init_state = 1
        if( self.bit == 32 ):
            return input
        else:
            # alpha = quan_alpha(self.alpha, 32)
            if self.alpha_bit == 32:
                alpha = self.alpha #
            else:
                # self.alpha_qfn(self.alpha)
                alpha = self.alpha
                q_code  = self.alpha_bit - torch.ceil( torch.log2( torch.max(alpha)) + 1 - 1e-5 )
                alpha = torch.clamp( torch.round( self.alpha * (2**q_code)), -2**(self.alpha_bit - 1), 2**(self.alpha_bit - 1) - 1 ) / (2**q_code)
            #     assert not torch.isinf(self.alpha).any(), self.alpha
            # assert not torch.isnan(input).any(), "Act_Q should not be 'nan'"
            act = self.round_fn.apply( input, alpha, self.pwr_coef, self.bit, self.signed)
            # assert not torch.isnan(act).any(), "Act_Q should not be 'nan'"
            return act
    def extra_repr(self):
        s_prefix = super(ACT_Q, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)



class Linear_Q(nn.Linear):
    def __init__(self,
                 in_features, 
                 out_features, 
                 bitW=32,
                 bias=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.bitW = bitW
        self.fw = fw(bitW)

    def forward(self, input):
        q_weight = self.fw(self.weight)
        return F.linear(input, q_weight, self.bias)

##########################################

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU()
        )

def conv_Q(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bitW=32, bitA=32):
    if batchNorm:
        return nn.Sequential(
            Conv2d_Q(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False, bitW=bitW),
            nn.BatchNorm2d(out_planes),
            Act_Q(bitA=bitA)
        )
    else:
        return nn.Sequential(
            Conv2d_Q(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True, bitW=bitW),
            nn.ReLU(),
            Act_Q(bit=bitA)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def predict_flow_Q(in_planes, bitW=32):
    return Conv2d_Q(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False, bitW=bitW)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU()
    )

def deconv_Q(in_planes, out_planes, bitW=32, bitA=32):
    return nn.Sequential(
        ConvTrans2d_Q(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False, bitW=bitW),
        nn.ReLU(),
        Act_Q(bit=bitA)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
