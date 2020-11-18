import os
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import sys
from torchsummary import summary


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):

    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def save_training_args(save_path, args):
    f = open(os.path.join(save_path,'training_parameter.txt'), "w")
    vars(args)
    for key, value in vars(args).items():
        f.write('{}\t\t{}'.format(key, value))
        f.write('\n')
    f.write('\n')
    f.write('\n{}'.format(' '.join(sys.argv)))
    f.close()

def exportpars(model, save_path, args):
    f = open(os.path.join(save_path,'{}_pars.txt'.format(args.arch)), "w")
    n = 0
    for key, val in model.module.state_dict().items():
        f.write ('{}, {}\t'.format(n, key))
        f.write ('{}\n'.format(val.size()))
        n+=1
    f.close()

def exportsummary(model, save_path, args):
    f = open(os.path.join(save_path,'{}_summary.txt'.format(args.arch)), "w")
    if args.grayscale:
        f.write ('{}\n'.format(summary(model.module, (2, 320, 448))))
    else:
        f.write ('{}\n'.format(summary(model.module, (6, 320, 448))))
    f.close()