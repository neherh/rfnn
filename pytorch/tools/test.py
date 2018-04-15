# test.py contains functions that were first tested here
import os, sys
from PIL import Image, ImageChops



def get_filenames(main_dir):

	#for loop directory
	list_of_img = []

	# walk through directories, get file info and create directories
	for (dirpath, dirnames, filenames) in os.walk(main_dir):

		# get image files and add to list
		for idx,filename in enumerate(filenames):
			if filename.endswith('.png'):
				filename = filename.split('_')
				# print(filename[0])
				list_of_img.append(os.sep.join([dirpath, filename[0]]))

	# make sure no duplicates, sort them and return
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
		print(name)
		print(name0)

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



# variables
index = 1
train_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_train/'
label_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_valid_pics/'

# In init
train_names = get_filenames(train_dir)


# in getItems
img0 = get_img0(train_names,index)
img1 = get_img1(train_names,index,img0.size)

# get labeled image target
label = get_imgT(train_names,index,train_dir,label_dir) # assume label for every input

# transform images


img3 = float(img0[:,:]/255)
# label.show()
# print(img0.size)
# print(img1.size)

# img0.show()
# img1.show()


# print(train_names)