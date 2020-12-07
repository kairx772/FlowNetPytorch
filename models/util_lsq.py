import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn import Parameter

class RoundFn_act(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit, signed):
        if signed == True:
            x_alpha_div = (input  / alpha ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) *  alpha
        else:
            x_alpha_div = (input  / alpha ).round().clamp( min =0, max = (pwr_coef-1)) *  alpha
        ctx.pwr_coef = pwr_coef
        ctx.bit      = bit
        ctx.signed   = signed
        ctx.save_for_backward(input, alpha)
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        pwr_coef = ctx.pwr_coef
        bit = ctx.bit
        signed = ctx.signed
       
        if signed == True:
            low_bound = -(pwr_coef)
        else:
            low_bound = 0
        quan_Em =  (input  / alpha   ).round().clamp( min =low_bound, max = (pwr_coef-1)) * alpha 
        quan_El =  (input / ( alpha  / 2)   ).round().clamp( min =low_bound, max = (pwr_coef-1)) * ( alpha  / 2)
        quan_Er = (input / ( alpha * 2)  ).round().clamp( min =low_bound, max = (pwr_coef-1)) * ( alpha * 2)
        El = torch.sum(torch.pow((input - quan_El), 2 ))
        Er = torch.sum(torch.pow((input - quan_Er), 2 ))
        Em = torch.sum(torch.pow((input - quan_Em), 2 ))
        d_better = torch.Tensor([El, Em, Er]).argmin() -1
        delta_G = (-1) * (torch.pow(alpha , 2)) * (  d_better) 

        #delta_G = alpha * (2**d_better)

        grad_input = grad_output.clone()
        # grad_input[grad_input!=grad_input] = 0
        if signed == True:
            # grad_input = torch.where((input) < ( (-1) * pwr_coef  * alpha ) , torch.full_like(grad_input,0), grad_input ) # ((-pwr_coef) * alpha)
            # grad_input = torch.where((input) > ((pwr_coef    - 1) * alpha ),  torch.full_like(grad_input,0), grad_input)
            grad_input[(input) < ( (-1) * pwr_coef  * alpha )] = 0
            grad_input[(input) > ((pwr_coef - 1) * alpha )] = 0

        else:
            # grad_input = torch.where( (input) < 0 , torch.full_like(grad_input,0), grad_input )
            # grad_input = torch.where((input) > ((pwr_coef - 1) * alpha ),  torch.full_like(grad_input,0), grad_input)
            grad_input[(input) < 0] = 0
            grad_input[(input) > ((pwr_coef - 1) * alpha )] = 0
            

        return  grad_input, delta_G, None, None, None

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
        # print ('self.alpha_bit: ', self.alpha_bit)
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

class RoundFn_LLSQ(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
        # the standard quantization function quantized to k bit, where 2^k=pwr_coef, the input must scale to [0,1]
        
        # alpha = quan_alpha(alpha, 16)
        x_alpha_div = (input  / alpha  ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha  
        
        ctx.pwr_coef = pwr_coef
        ctx.bit      = bit
        ctx.save_for_backward(input, alpha)
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        pwr_coef = ctx.pwr_coef
        bit      = ctx.bit
        #alpha = quan_alpha(alpha, 16)
        quan_Em =  (input  / (alpha ) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha  
        quan_El =  (input / ((alpha ) / 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  / 2) 
        quan_Er = (input / ((alpha ) * 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  * 2) 
        
        if list(alpha.size())[0] > 1:
            El = torch.sum(torch.pow((input - quan_El), 2 ), dim = 0)
            Er = torch.sum(torch.pow((input - quan_Er), 2 ), dim = 0)
            Em = torch.sum(torch.pow((input - quan_Em), 2 ), dim = 0)
            
            d_better = torch.argmin( torch.stack([El, Em, Er], dim=0), dim=0) -1
            delta_G = - (torch.pow(alpha , 2)) * ( d_better)
        else:
            El = torch.sum(torch.pow((input - quan_El), 2 ))
            Er = torch.sum(torch.pow((input - quan_Er), 2 ))
            Em = torch.sum(torch.pow((input - quan_Em), 2 ))
            d_better = torch.Tensor([El, Em, Er]).argmin() -1
            delta_G = (-1) * (torch.pow(alpha , 2)) * ( d_better) 
            
        #delta_G = alpha * (2**d_better)


        grad_input = grad_output.clone()
        return  grad_input, delta_G, None, None


class RoundFn_Bias(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
        ctx.save_for_backward(input, alpha)
        # alpha = quan_alpha(alpha, 16)
        alpha = torch.reshape(alpha, (-1,))
        # alpha = quan_alpha(alpha, bit)
        x_alpha_div = (input  / alpha).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha 
        ctx.pwr_coef = pwr_coef
        
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return  grad_input, None, None, None


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        #self.alpha_w = Variable(torch.rand( out_channels,1,1,1)).cuda()
        # self.alpha_w = Parameter(torch.rand( out_channels))
        if bit < 0:
            self.alpha_w = None
        else:
            self.alpha_w = Parameter(torch.rand( out_channels))
        #self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        if bit < 0:
            self.init_state = 0
        else:
            self.register_buffer('init_state', torch.zeros(1))
        # self.init_state = 0
    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:
            w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
            if self.training and self.init_state == 0:            
                self.alpha_w.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / self.pwr_coef)
                self.init_state.fill_(1)
                #self.init_state = 1

            #assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            alpha_w = self.alpha_w #self.alpha_qfn(self.alpha_w)
            #if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
            #    assert not torch.isnan(wq).any(), self.alpha_w
            #    assert not torch.isinf(wq).any(), self.alpha_w

            wq =  self.Round_w(w_reshape, alpha_w, self.pwr_coef, self.bit)
            w_q = wq.transpose(0, 1).reshape(self.weight.shape)

            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            # assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            # if torch.isnan(wq).any() or torch.isinf(wq).any():
            #     print(self.alpha_w)
            #     assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
            #     assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            
            return F.conv2d(
                x,  w_q, LLSQ_b, self.stride, self.padding, self.dilation,
                self.groups)
    def extra_repr(self):
        s_prefix = super(QuantConv2d, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)

class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        #self.alpha_w = Variable(torch.rand( out_channels,1,1,1)).cuda()
        # self.alpha_w = Parameter(torch.rand( out_channels))
        if bit < 0:
            self.alpha_w = None
        else:
            self.alpha_w = Parameter(torch.rand( out_channels))
        #self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        if bit < 0:
            self.init_state = 0
        else:
            self.register_buffer('init_state', torch.zeros(1))
        # self.init_state = 0
    def forward(self, x):
        if self.bit == 32:
            return F.conv_transpose2d(
                input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups)
        else:
            w_reshape = self.weight.reshape([self.weight.shape[1], -1]).transpose(0, 1)
            if self.training and self.init_state == 0:            
                self.alpha_w.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / self.pwr_coef)
                self.init_state.fill_(1)
                #self.init_state = 1

            #assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            alpha_w = self.alpha_w #self.alpha_qfn(self.alpha_w)
            #if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
            #    assert not torch.isnan(wq).any(), self.alpha_w
            #    assert not torch.isinf(wq).any(), self.alpha_w

            wq =  self.Round_w(w_reshape, alpha_w, self.pwr_coef, self.bit)
            w_q = wq.transpose(0, 1).reshape(self.weight.shape)

            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            # assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            # if torch.isnan(wq).any() or torch.isinf(wq).any():
            #     print(self.alpha_w)
            #     assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
            #     assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            
            return F.conv_transpose2d(
                input=x, weight=w_q, bias=LLSQ_b, stride=self.stride, padding=self.padding, dilation=self.dilation,
                groups=self.groups)
    def extra_repr(self):
        s_prefix = super(QuantConvTranspose2d, self).extra_repr()
        if self.alpha_w is None:
            return '{}, fake'.format(s_prefix)
        return '{}'.format(s_prefix)


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

def conv_Q(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bitW=32, bitA=32, bias=True):
    if batchNorm:
        return nn.Sequential(
            QuantConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False, bit = bitW),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            ACT_Q(bit=bitA)
        )
    else:
        return nn.Sequential(
            QuantConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias, bit = bitW),
            nn.ReLU(),
            ACT_Q(bit=bitA)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def predict_flow_Q(in_planes, bitW=32):
    return QuantConv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False, bit=bitW)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU()
    )

def deconv_Q(in_planes, out_planes, bitW=32, bitA=32):
    return nn.Sequential(
        QuantConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False, bit=bitW),
        nn.ReLU(),
        ACT_Q(bit=bitA)
    )

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]