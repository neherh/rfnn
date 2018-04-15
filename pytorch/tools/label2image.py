# Label to Image
# grabs the labels from txt format and saves the lane detections
# as a mono image.

# assumptions: 
#	- assume that the new_dir exists by doesn't have any other directories existing
# 	- 

import os, sys
import numpy as np
import cv2
# from PIL import Image

# variables
main_dir = '/media/vidavilane/External Drive/dataSets/cuLane_SCNN_Results/label/'
new_dir = '/media/vidavilane/External Drive/dataSets/cuLane_SCNN_Results/label_pics/'

n = 1640 # x (column)
m = 590  # y (row)

img = np.zeros((m,n,1),np.float32)

# go to big directory and determine any other directories exist

#for loop directory
list_of_txt = {}
list_of_img = {}
idx_txt = 0;
idx_img = 0;


# walk through directories, get file info and create directories
for (dirpath, dirnames, filenames) in os.walk(main_dir):
	# if there are filenames in this directory
	if filenames:
		# print("full")
		# print(dirpath)
		# get newPath and create directories
		newPath = dirpath.replace(main_dir,new_dir)

		if not os.path.exists(os.path.dirname(newPath + '/')):
		    try:
		        os.makedirs(os.path.dirname(newPath + '/'))
		    except OSError as exc: # Guard against race condition
		        if exc.errno != errno.EEXIST:
		            raise

		# create
		# print(newPath)
	# else:
	# 	print("empty")


	# separate files into txt and jpg
	for filename in filenames:
		if filename.endswith('.txt'): 
			list_of_txt[os.sep.join([dirpath, filename])] = os.sep.join([dirpath, filename])
		if filename.endswith('.jpg'):
			list_of_img[os.sep.join([dirpath, filename])] = os.sep.join([dirpath, filename])

# cycle through text files, get info, and then create a single mono image
for key in list_of_txt:
	print(key)


	# init new image
	image = np.zeros((m,n,1),np.float32)

	
	# # open file and get num of lines
	f = open(list_of_txt[key],'r')
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

		# print(x[0])
		points = np.transpose(np.array([x,y], np.float32))
		print(points)
		# print(points[:,0])
		# # print(points)
		# cv2.circle(image,(int(x[0]),int(y[0])),11,(255),1) 
		cv2.polylines(image, np.int32([points]), False, 255,3)

	img = image
	imgPath = key.replace(main_dir,new_dir)
	imgPath = imgPath.split('.lines.txt')
	imgPath = imgPath[0] + '.png'
	print(imgPath)

	# cv2.imwrite(imgPath,img)


winname = 'example'
img2 = cv2.resize(img,(200,200))
cv2.namedWindow(winname)
cv2.imshow(winname, img2)
cv2.waitKey()
cv2.destroyWindow(winname)





# determine number of lines


# assume all images of 1640 x 590
