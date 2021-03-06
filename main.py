import argparse
import os
import time
import sys
import json
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
from multiscaleloss import multiscaleEPE, realEPE
import datetime
from torch.utils.tensorboard import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint, save_training_args, exportpars, exportsummary
import numpy as np
import cv2
import flow_viz

# import warnings
# warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', 
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
parser.add_argument('--solver', default='adamw',choices=['adam','sgd', 'adamw'],
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
parser.add_argument('--gamma', default=0.5, type=float, metavar='M',
                    help='gamma for optimizer scheduler')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.32,0.08,0.001,0.001,0.001], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6')) # 0.005,0.01,0.02,0.08,0.32
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--grayscale', action='store_true',
                    help='convert dataset to grayscale' )
parser.add_argument('--graymodel', action='store_true',
                    help='use gray scale model' )
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-demo', '--demo', dest='demo', action='store_true',
                    help='demo model on webcam')
parser.add_argument('-demovideo', '--demovideo', dest='demovideo', action='store_true',
                    help='demo model for video')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--savpath', default=None, type=str,
                    help='folder path name' )
parser.add_argument('--print-model', action='store_true',
                    help='print model parameters' )
parser.add_argument('--div-flow', default=20, type=int,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--qw', default=None, type=int,
                    help='weight quantization')
parser.add_argument('--qa', default=None, type=int,
                    help='activation quantization')
parser.add_argument('--cut-ratio', default=2, type=int,
                    help='divide FlowNet channels number by a ratio')
parser.add_argument('--alphabit', default=None, type=int,
                    help='alpha_bit for LLSQ quantization')
parser.add_argument('--conv-no-bias', action='store_false',
                    help='convolution layer have no bias parameters' )
parser.add_argument('--milestones', type=int, default=[100,150,200], metavar='N', nargs='+', help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main():
    global args, best_EPE
    args = parser.parse_args()
    
    if not args.data:
        f = open('train_src/data_loc.json', 'r')
        content = f.read()
        f.close()
        data_loc = json.loads(content)
        args.data = data_loc[args.dataset]
    
    if not args.savpath:
        save_path = '{},{},{}epochs{},b{},lr{}'.format(
            args.arch,
            args.solver,
            args.epochs,
            ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
            args.batch_size,
            args.lr)
        if not args.no_date:
            timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
            save_path = os.path.join(timestamp,save_path)
    else:
        save_path = args.savpath
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save training args
    save_training_args(save_path, args)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    if args.grayscale:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.431,0.431,0.431], std=[1,1,1]) # 0.431=(0.45+0.432+0.411)/3
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[1,1,1])
        ])
    else:
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
    if not args.evaluate:
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

    # if (args.qw and args.qa and args.cut_ratio) is not None:
    #     model = models.__dict__[args.arch](data=network_data, bitW=args.qw, bitA=args.qa, cut_ratio=args.cut_ratio).cuda()
    # elif (args.qw and args.qa) is not None:
    #     model = models.__dict__[args.arch](data=network_data, bitW=args.qw, bitA=args.qa).cuda()
    # else:
    #     model = models.__dict__[args.arch](data=network_data).cuda()

    model = models.__dict__[args.arch](data=network_data, args=args).to(device)

    # model = torch.nn.DataParallel(model).cuda()
    # cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd', 'adamw'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)
    elif args.solver == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, args.lr,
                                    betas=(args.momentum, args.beta))
    
    if args.print_model:
        exportpars(model, save_path, args)
        exportsummary(model, save_path, args)
        if args.savpath == 'test':
            return

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    if args.demo:
        demo(val_loader, model, 0, output_writers)
        return
    if args.demovideo:
        demovideo(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)
        scheduler.step()

        # evaluate on validation set
        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best, save_path)


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)

        # concatnate the tensor
        input = torch.cat(input,1).to(device)

        # compute output
        if args.graymodel:
            output = model(torch.cat([input[:,0:1,:,:], input[:,3:4,:,:]],1))
        else:
            output = model(input)
            # print ('input : ', input[0].shape)
            # print ('output: ', output[0].shape)
        # torch.onnx.export(model.module, torch.cat([input[:,0:1,:,:], input[:,3:4,:,:]],1), "model.onnx", opset_version=11, verbose=True)

        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h,w)), *output[1:]]

        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time(); 
    runtime_count = 0; 
    sum_runtime = 0;
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        # concatnate the tensor
        input = torch.cat(input,1).to(device)
        
        # compute output
        start_time = time.time()
        if args.graymodel:
            output = model(torch.cat([input[:,0:1,:,:], input[:,3:4,:,:]],1))
        else:
            output = model(input)
        runtime = (time.time()-start_time)
        #sum_runtime += runtime
        # if runtime_count == 0: 
        #     sum_runtime = 0 
        # else: print ('AvgRunTime: ', sum_runtime/runtime_count)
        runtime_count += 1
        sum_runtime += runtime
        # print (runtime)
        flow2_EPE = args.div_flow*realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
                output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))
    print ('sum_runtime', sum_runtime/runtime_count, runtime_count)

    return flow2_EPEs.avg

def demo(val_loader, model, epoch, output_writers):
    global args

    # switch to evaluate mode
    model.eval()

    cap = cv2.VideoCapture(0)
    ret, image1 = cap.read()
    if args.graymodel:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    loader = transforms.Compose([transforms.ToTensor()])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    while(True):
        # Capture frame-by-frame
        ret, image2 = cap.read()
        if args.graymodel:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image1cu = loader(image1).unsqueeze(0).to(device, torch.float)
        image2cu = loader(image2).unsqueeze(0).to(device, torch.float)

        output = args.div_flow * model(torch.cat([image1cu, image2cu],1))
        output = F.interpolate(output, size=image1cu.size()[-2:], mode='bilinear', align_corners=False)
        # flow_low, flow_pr = model(image1cu, image2cu, iters=iters, test_mode=True)
        # flow = padder.unpad(flow_pr[0]).cpu()
        
        flo = output[0].permute(1,2,0).cpu().detach().numpy()



        flo = flow_viz.flow_to_image(flo)
        flow = flo/255.0

        cv2.imshow('frame', image2)
        cv2.imshow('OF', flo)
        # print (image2.shape, flo.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        image1 = image2

    return

def demovideo(val_loader, model, epoch, output_writers):
    model.eval()
    cap = cv2.VideoCapture('train_src/video/2011_09_30_drive_0027_sync_Image.mkv')
    ret, image1 = cap.read()
    loader = transforms.Compose([transforms.ToTensor()])
    fnum = 0
    while(True):
        ret, image2 = cap.read()
        image1cu = loader(image1).unsqueeze(0).to(device, torch.float)
        image2cu = loader(image2).unsqueeze(0).to(device, torch.float)
        output = args.div_flow * model(torch.cat([image1cu, image2cu],1))
        output = F.interpolate(output, size=image1cu.size()[-2:], mode='bilinear', align_corners=False)
        fnum += 1
        print (fnum)
        flo = output[0].permute(1,2,0).cpu().detach().numpy()
        np.save('train_src/video/2011_09_30_drive_0027_sync_Image/flownets_w5a5/{}'.format(fnum), flo)
    cap.release()

if __name__ == '__main__':
    main()