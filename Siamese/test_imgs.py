import torch
import os

from model import SiameseNet
from utils import get_num_model

def load_checkpoint(model, ckpt_dir, best=False):
    print("[*] Loading model from {}".format(ckpt_dir))
    filename = 'model_ckpt.tar'
    if best:
        filename = 'best_model_ckpt.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state'])


def main():
    ckpt_dir = './ckpt/exp_1/'
    model = SiameseNet()
    model.cuda()
    load_checkpoint(model, ckpt_dir, best=False)



if __name__ == '__main__':
    main()
