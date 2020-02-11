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
from model import _G_xvz, _G_vzx
from itertools import *
from dream_model import load_face_model
import pdb

dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str, default="./list_test_complete.txt")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument('--outf', default='./evaluate/SP', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./pretrained_model/VS', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
save_images = False

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_xvz = _G_xvz()
G_vzx = _G_vzx()

train_list = args.data_list

train_loader = torch.utils.data.DataLoader(
    data_loader_evaluate.ImageList( train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x-y), 1))


x = torch.FloatTensor(args.batch_size, 3, 128, 128)
x_bar_bar_out = torch.FloatTensor(10, 3, 128, 128)

v_siz = 9
z_siz = 128-v_siz
v = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_xvz = torch.nn.DataParallel(G_xvz).cuda()
    G_vzx = torch.nn.DataParallel(G_vzx).cuda()

    x = x.cuda()
    x_bar_bar_out = x_bar_bar_out.cuda()
    v = v.cuda()
    z = z.cuda()

x = Variable(x)
x_bar_bar_out = Variable(x_bar_bar_out)
v = Variable(v)
z = Variable(z)

def load_pretrained_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path,name))
    own_state = net.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)

load_pretrained_model(G_xvz, args.modelf, 'netG_xvz.pth')
load_pretrained_model(G_vzx, args.modelf, 'netG_vzx.pth')

batch_size = args.batch_size
cudnn.benchmark = True
G_xvz.eval()
G_vzx.eval()

error = []
face_model = load_face_model()
# face_model = _G_xvz()
# face_model = torch.nn.DataParallel(face_model).cuda()
# load_pretrained_model(G_xvz, './pretrained_model/no_selection', 'netG_xvz.pth')

error_tensor = []
for i, (data) in enumerate(train_loader):
    print i,
    img = data
    x.data.resize_(img.size()).copy_(img)

    x_bar_bar_out.data.zero_()
    v_bar, z_bar = G_xvz(x)
    x_bar = G_vzx(v_bar, z_bar)

    feature_x = face_model(data.cuda(), 0)
    # _, feature_x = face_model(data.cuda())
    feature_x = feature_x.cpu().data.numpy().ravel()

    feature_x_bar = face_model(x_bar, 0)
    # _,feature_x_bar = face_model(x_bar)
    feature_x_bar = feature_x_bar.cpu().data.numpy().ravel()
    # feature_x_bar = np.random.randn(feature_x.shape[0])
    error.append(np.linalg.norm(feature_x - feature_x_bar)/np.linalg.norm(feature_x))
    # error.append(np.linalg.norm(feature_x - feature_x_bar))
    print np.linalg.norm(feature_x - feature_x_bar),
    print ' average error: ', np.mean(error), '+-', np.std(error)
    if save_images:
        for one_view in range(9):
            v.data.zero_()
            for d in range(data.size(0)):
                v.data[d][one_view] = 1
            exec('x_bar_bar_%d = G_vzx(v, z_bar)' % (one_view))

        for d in range(batch_size):
            x_bar_bar_out.data[0] = x.data[d]
            for one_view in range(9):
                exec('x_bar_bar_out.data[1+one_view] = x_bar_bar_%d.data[d]' % (one_view))

            vutils.save_image(x_bar_bar_out.data,
                        '%s/%d_x_bar_bar.png' % (args.outf, i*batch_size+d), nrow = 10, normalize=True, pad_value=255)

if train_list.__contains__('complete'):
    error_tensor = np.array(error).reshape([49, 2, 9, 20])
    # print 'mean for angles'
    print(np.mean(error_tensor, (0, 1, 3)))
    # print 'mean for illuminations'
    print(np.mean(error_tensor, (0, 1, 2)))
    np.save(args.outf + '/error_tensor.npy', error_tensor)
print 'Done!'
print(args)
