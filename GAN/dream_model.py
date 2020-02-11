import argparse
import os, sys, shutil
import time
import struct as st

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from dream.ResNet import resnet18, resnet50, resnet101

def load_face_model():
    arch = 'resnet18_naive'
    model_path = './dream/ijba_res18_naive.pth.tar'
    yaw_type = 'nonli'

    global args, best_prec1

    if arch.find('end2end') >= 0:
        end2end = True
    else:
        end2end = False

    arch = arch.split('_')[0]

    class_num = 87020
    # class_num = 13386

    model = None
    assert (arch in ['resnet18', 'resnet50', 'resnet101'])
    if arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, \
                         extract_feature=True, end2end=end2end)
    if arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, \
                         extract_feature=True, end2end=end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num, \
                          extract_feature=True, end2end=end2end)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    assert (os.path.isfile(model_path))
    checkpoint = torch.load(model_path)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict)

    print('load trained model complete')

    return model