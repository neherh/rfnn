# Label to Image
# grabs the labels from txt format and saves the lane detections
# as a mono image.

# assumptions: 
#	- assume that the new_dir exists by doesn't have any other directories existing
# 	- 

import os, sys
import numpy as np
import cv2
from time import sleep
# from PIL import Image

# variables
# main_dir = '/media/vidavilane/External Drive/dataSets/cuLane_SCNN_Results/label/'
# new_dir = '/media/vidavilane/External Drive/dataSets/cuLane_SCNN_Results/label_pics/'

train_dir = '/home/vidavilane/Documents/datasets/cuLane_SCNN_results/train/'
test_dir  = '/home/vidavilane/Documents/datasets/cuLane_SCNN_results/test/'
main_dir = '/home/vidavilane/Documents/datasets/cuLane_SCNN_results/label/'
new_dir = '/home/vidavilane/Documents/datasets/cuLane_SCNN_results/label_pics2/'

n = 1640 # x (column)
m = 590  # y (row)

img = np.zeros((m,n,1),np.float32)

# go to big directory and determine any other directories exist

#for loop directory
list_of_txt = {}
list_of_img = {}
idx_txt = 0;
idx_img = 0;


# walks through directories and add to filenames
def get_filenames(main_dir):
    list_of_img = []

    # walk through directories, get file info and create directories
    for (dirpath, dirnames, filenames) in os.walk(main_dir):

        # get image files and add to list
        for idx,filename in enumerate(filenames):
            if filename.endswith('.png'):
            	filename = filename.split('_')
            	list_of_img.append(os.sep.join([dirpath, filename[0]])) # labels output with .png, other do not

    list_of_img = list(set(list_of_img))
    list_of_img.sort()
    return list_of_img

# get file names from test and train
train_files = get_filenames(train_dir)
test_files = get_filenames(test_dir)

# concatenate and change to label file path
train_files = [w.replace(train_dir,main_dir) for w in train_files]
test_files = [w.replace(test_dir,main_dir) for w in test_files]

label_files = train_files + test_files
label_files.sort()



for w in label_files:

	# get the new image file path and get the txt_files
	txt_file = w + '.lines.txt'
	img_file = w.replace(main_dir,new_dir) + '.png'

	# create new directory (add any folders if needed)
	if not os.path.exists(os.path.dirname(img_file)):
            try:
                os.makedirs(os.path.dirname(img_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

	# print(txt_file)
	# print(img_file)


	# init new image
	image = np.zeros((m,n,1),np.float32)

	
	# # open file and get num of lines
	f = open(txt_file,'r')
	f_line = f.readlines()

	list_length = len(f_line)

	# for each line
	for line in f_line:
		# if
		# print(line)

		# split line to points, then assign x and y val
		splitLine = line.split(' ')
		x = splitLine[0:len(splitLine) - 1:2]
		y = splitLine[1:len(splitLine) - 1:2]

		# convert str to float
		x = list(map(float,x))
		y = list(map(float,y))

		points = np.transpose(np.array([x,y], np.float32))

		cv2.polylines(image, np.int32([points]), False, 255,30)

	img = image

	cv2.imwrite(img_file,cv2.bitwise_not(img))
	sleep(.1)
	# idx_txt += 1

	# if (idx_txt % 5) == 0:
	# 	print(idx_txt)

		# winname = 'example'
# img2 = cv2.resize(img,(200,200))
# cv2.namedWindow(winname)
# cv2.imshow(winname, img2)
# cv2.waitKey()
# cv2.destroyWindow(winname)



