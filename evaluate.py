import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import models
import datasets
from datasets import datasets_f2
from multiscaleloss import multiscaleEPE, realEPE
import datetime
from tensorboardX import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint, InputPadder

import warnings
warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--qw', default=None, type=int,
                    help='weight quantization')
parser.add_argument('--qa', default=None, type=int,
                    help='activation quantization')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main():
    global args, best_EPE
    args = parser.parse_args()

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    if 'KITTI' in args.dataset:
        args.sparse = True
    if args.sparse:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])
    else:
        co_transform = flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10,5),
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samp-les found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        # args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    if args.qw and args.qa is not None:
        model = models.__dict__[args.arch](data=network_data, bitW=args.qw, bitA=args.qa).cuda()
    else:
        model = models.__dict__[args.arch](data=network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0)
        # validate_chairs(model.module)
        # validate_sintel(model.module)
        return
    





def validate(val_loader, model, epoch):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time(); runtime_count = -1; sum_runtime = 0;
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        # concatnate the tensor
        input = torch.cat(input,1).to(device)
        
        # compute output
        # start_time = time.time(); runtime_count += 1;
        output = model(input)
        # runtime = (time.time()-start_time); sum_runtime += runtime
        # if runtime_count == 0: 
        #     sum_runtime = 0 
        # else: print ('AvgRunTime: ', sum_runtime/runtime_count)
        # print ('RunTime:    ', runtime)
        flow2_EPE = args.div_flow*realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        '''
        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
                output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)
        '''

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg

@torch.no_grad()
def validate_sintel(model):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    end_time = time.time()
    time_count = 0
    sum_time = 0
    for dstype in ['clean', 'final']:
        val_dataset = datasets_f2.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            start_time = time.time()

            input = torch.cat((image1, image2), 1).to(device)
            output = model(input)

            target = flow_gt.to(device)
            _, h, w = target.size()

            # print ('output size:', output.size())
            # print ('target size:', target.size())
            flow_pr = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
            # flow_pr = realEPE(output, target, sparse=args.sparse)
            
            # flow_pr = model(image1, image2)
            sum_time += (time.time()-start_time)
            time_count += 1
            # print ('MeanRunTime: ', (sum_time/time_count))
            # print ('EpoTime: ', (time.time()-end_time))
            end_time = time.time()

            flow = (flow_pr).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print ('MeanRunTime: ', (sum_time/time_count))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_chairs(model):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    end_time = time.time()
    val_dataset = datasets_f2.FlyingChairs(split='validation')
    time_count = 0
    sum_time = 0
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        input = torch.cat((image1, image2), 1).to(device)
        output = model(input)

        target = flow_gt.to(device)
        _, h, w = target.size()

        flow_pr = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)

        start_time = time.time()
        # _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        sum_time += (time.time()-start_time)
        time_count += 1
        print ('MeanRunTime: ', (sum_time/time_count))
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        # print ('EpoTime: ', (time.time()-end_time))
        end_time = time.time()

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

if __name__ == '__main__':
    main()