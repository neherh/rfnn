''' 
	TestResults.py
		Loads saved model of choice, gets output and places in specified folder and path
'''

#Imports
from __future__ import print_function
import argparse
from math import log10

import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net
from dataset import DatasetFromFolder # home-brew from file dataset.py
# from data import get_set # home-brew file (data.py)

# for saving images because we need to create directories)
import os
import errno 

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
# parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

# variables
pretr_model = '/home/vidavilane/Documents/repos/me640/pytorch/model_epoch_5.pth'
cuda = opt.cuda


print('===> Loading datasets')
test_dir  = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_test/'
label_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_valid_pics/'
preds_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_preds/'
ratio = 3 # must be int # res*ratio = width. this maintains ratio of height and width
res = 100

test_set = DatasetFromFolder(test_dir, label_dir,
                             transform=transforms.Compose([transforms.Resize((res,res*ratio)), # no angle changes
                                                                      transforms.ToTensor()
                                                                      ]))

# place lists into the dataloader
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> loading model')
model = torch.load(pretr_model)        # load model


print('===> evaluating')

def test():
    # local variables
    name_idx = 0

    for batch in testing_data_loader:
        img0,img1, target = Variable(batch[0]), Variable(batch[1]),  Variable(batch[2])
        input = torch.cat([img0,img1],1) # concatenate from (4,1,300,100) to (4,2,300,100) // (batchSize,Depth, Width, Height)

        if cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)

        name = test_set.get_labelPath(name_idx)
        name = name.replace(label_dir,preds_dir)

        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        torchvision.utils.save_image(prediction.data,name)
        name_idx += 1

test()