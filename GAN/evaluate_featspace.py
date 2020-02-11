import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import data_loader_evaluate
from torch.autograd import Variable
from model import _G_xvz, _G_vzx, z_siz
from itertools import *
import pdb

dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str, default="./list_test.txt")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument('--outf', default='./evaluate/feature/9', help='folder to output images and model checkpoints')
# parser.add_argument('--modelf', default='./pretrained_model/Random72',
#                     help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_xvz_random = _G_xvz()
G_xvz_SP = _G_xvz()
G_vzx_random = _G_vzx()
G_vzx_SP = _G_vzx()

train_list = args.data_list

train_loader = torch.utils.data.DataLoader(
    data_loader_evaluate.ImageList(train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x - y), 1))


x = torch.FloatTensor(args.batch_size, 3, 128, 128)
x_bar_bar_out = torch.FloatTensor(20, 3, 128, 128)

v_siz = 9
z_siz = 128 - v_siz
v = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_xvz_random = torch.nn.DataParallel(G_xvz_random).cuda()
    G_xvz_SP  = torch.nn.DataParallel(G_xvz_SP).cuda()
    G_vzx_random = torch.nn.DataParallel(G_vzx_random).cuda()
    G_vzx_SP  = torch.nn.DataParallel(G_vzx_SP).cuda()

    x = x.cuda()
    x_bar_bar_out = x_bar_bar_out.cuda()
    v = v.cuda()
    z = z.cuda()

x = Variable(x)
x_bar_bar_out = Variable(x_bar_bar_out)
v = Variable(v)
z = Variable(z)


def load_pretrained_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path, name))
    own_state = net.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)


load_pretrained_model(G_xvz_random, './output/Random9', 'netG_xvz_epoch_299_49.pth')
load_pretrained_model(G_xvz_SP, './output/SP9', 'netG_xvz_epoch_299_49.pth')
load_pretrained_model(G_vzx_SP, './output/Random9', 'netG_vzx_epoch_299_49.pth')
load_pretrained_model(G_vzx_SP, './output/SP9', 'netG_vzx_epoch_299_49.pth')

batch_size = args.batch_size
cudnn.benchmark = True
G_xvz_random.eval()
G_xvz_SP.eval()
G_vzx_random.eval()
G_vzx_SP.eval()

sum_error_random = 0
sum_error_SP  = 0
for i, (data) in enumerate(train_loader):
    print i,
    img = data
    x.data.resize_(img.size()).copy_(img)

    x_bar_bar_out.data.zero_()
    v_bar, z_bar = G_xvz_random(x)
    x_bar = G_vzx_random(v_bar, z_bar)
    sum_error_random += torch.norm(x_bar.cpu() - data) / torch.norm(data)
    print 'random: ', sum_error_random/(i + 1),
    for one_view in range(9):
        v.data.zero_()
        for d in range(data.size(0)):
            v.data[d][one_view] = 1
        exec ('x_bar_bar_%d = G_vzx_random(v, z_bar)' % (one_view))

    for d in range(batch_size):
        x_bar_bar_out.data[0] = x.data[d]
        for one_view in range(9):
            exec ('x_bar_bar_out.data[one_view+1] = x_bar_bar_%d.data[d]' % (one_view))

    v_bar, z_bar = G_xvz_SP(x)
    x_bar = G_vzx_SP(v_bar, z_bar)
    sum_error_SP += torch.norm(x_bar.cpu() - data) / torch.norm(data)
    print 'SP: ', sum_error_SP/(i + 1)

    for one_view in range(9):
        v.data.zero_()
        for d in range(data.size(0)):
            v.data[d][one_view] = 1
        exec ('x_bar_bar_%d = G_vzx_SP(v, z_bar)' % (one_view))

    for d in range(batch_size):
        x_bar_bar_out.data[10] = x.data[d]
        for one_view in range(9):
            exec ('x_bar_bar_out.data[one_view + 11] = x_bar_bar_%d.data[d]' % (one_view))

        vutils.save_image(x_bar_bar_out.data,
                          '%s/%d_x_bar_bar.png' % (args.outf, i * batch_size + d), nrow=10, normalize=True,
                          pad_value=255)
