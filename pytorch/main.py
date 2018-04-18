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
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    print('using cuda')
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_dir = '/home/neherh/cuLane_SCNN_Results/train/'
test_dir  = '/home/neherh/cuLane_SCNN_Results/test/'
label_dir = '/home/neherh/cuLane_SCNN_Results/label_pics2/'
preds_dir = '/home/neherh/cuLane_SCNN_Results/preds2/'

#train_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_train/'
#test_dir  = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_test/'
#label_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_valid_pics/'
#preds_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_preds/'
ratio = 3 # must be int # res*ratio = width. this maintains ratio of height and width
res = 256

# get train and test set (list of data, length and how to get items)
train_set = DatasetFromFolder(train_dir, label_dir,
                             transform=transforms.Compose([transforms.Resize((res,res*ratio)), # no angle changes
                                                                      transforms.ToTensor(),
                                                                      ]))
test_set = DatasetFromFolder(test_dir, label_dir,
                             transform=transforms.Compose([transforms.Resize((res,res*ratio)), # no angle changes
                                                                      transforms.ToTensor()
                                                                      ]))

# place lists into the dataloader
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net()#upscale_factor=opt.upscale_factor)
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss2d(size_average=True)
criterion = nn.BCEWithLogitsLoss()
# c = nn.LogSoftmax()

# criterion = nn.MSELoss()


if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',0.1,3) 



def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        img0,img1,target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        # print("type is:")
        # print(img0.data.size())
        # print(img0.data.type())
        input = torch.cat([img0,img1],1) # concatenate from (4,1,300,100) to (4,2,300,100) // (batchSize,Depth, Width, Height)
        
        # input = input.type(torch.LongTensor)
        # target = target.type(torch.LongTensor)
        # print(val.data.size())
        if cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        optimizer.zero_grad()
        # loss = criterion(c(output), target.view(-1)
        loss = criterion(output, target)

        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    scheduler.step(epoch_loss / len(training_data_loader))

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


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# variables
counter = 0

# run script
for epoch in range(1, opt.nEpochs + 1):
    
    # train 
    train(epoch)

    # save model every 5 epochs
    counter += 1
    print(counter)
    if counter == 5 or epoch == opt.nEpochs:
        checkpoint(epoch)
        counter = 0

# at end, run through test code and save the images somewhere
test()
