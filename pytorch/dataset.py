import torch.utils.data as data
import torchvision
import os, sys
from os import listdir
from os.path import join
from PIL import Image,ImageChops

'''
To do:
    - __get_item__
Done:
    - get_fileNames
    - __len__ 
'''#################

# walks through directories and add to filenames
def get_filenames(main_dir):
    list_of_img = []

    # walk through directories, get file info and create directories
    for (dirpath, dirnames, filenames) in os.walk(main_dir):

        # get image files and add to list
        for idx,filename in enumerate(filenames):
            if filename.endswith('.png'):
                filename = filename.split('_')
                # print(filename[0])
                list_of_img.append(os.sep.join([dirpath, filename[0]])) # labels output with .png, other do not

    list_of_img = list(set(list_of_img))
    list_of_img.sort()
    return list_of_img

def get_img0(names,index):
    # parse data and get all images
    # print(names[index] + '_1_.png')
    img1 =Image.open(names[index] + '_1_avg.png')
    img2 =Image.open(names[index] + '_2_avg.png')
    img3 =Image.open(names[index] + '_3_avg.png')
    img4 =Image.open(names[index] + '_4_avg.png')


    img0 = ImageChops.add(img1,img2)
    img0 = ImageChops.add(img0,img4)
    img0 = ImageChops.add(img0,img3)

    # img0.show()

    return img0

def get_img1(names,index,size):
    # parse data and get all images
    if index == 0:
        return Image.new('L',size)
    else:
        name = names[index]
        name = name.split('/')
        name = name[len(name)-1]

        name0 = names[index-1]
        name0 = name0.split('/')
        name0 = name0[len(name0)-1]
        # print(name)
        # print(name0)

        # create blank image if first in sequence, else get img
        if name <= name0:
            return Image.new('L',size)
        else:
            img1 = get_img0(names,index - 1)
            return img1


def get_imgT(names,index,train_dir,label_dir):
    labelPath = names[index].replace(train_dir,label_dir)
    # print(labelPath)
    target =Image.open(labelPath + '.png')

    return target


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,label_dir, transform=None):
        super(DatasetFromFolder, self).__init__()

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = get_filenames(self.image_dir)  # in a list without _x_.png ext
        # self.label_filenames = get_filenames(self.label_dir) # in a list with png ext
        self.transform = transform

    def __getitem__(self, index):

        # get input
        img0 = get_img0(self.image_filenames,index)
        img1 = get_img1(self.image_filenames,index,img0.size)

        # get labeled image target
        label = get_imgT(self.image_filenames,index,self.image_dir,self.label_dir)

        # transform if needed
        if self.transform is not None:
            img0  = self.transform(img0)
            img1  = self.transform(img1)
            label = self.transform(label)

        return img0, img1, label

    def __len__(self):
        return len(self.image_filenames)

    def get_labelPath(self,index):
        return self.image_filenames[index].replace(self.image_dir,self.label_dir) + '.png'
