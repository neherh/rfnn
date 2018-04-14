######################################################################
# imageOverlay.py
# This file is for the tuLane dataset. It overlays the estimated lane
# detection over the actual lane detection
######################################################################

# test images from dataset are:
# 120, 150, 180, 210, 240, 270, 300, 330

import matplotlib.pyplot as plt
import numpy as np
# declare variables
imgDir  = '/home/jzelek/Documents/datasets/CULane/driver_37_30frame/05181432_0203.MP4/'
imgFile = '02700.lines.txt'
imgName = '02700.jpg'
imgDirEst = '/home/jzelek/Documents/repos/ME640/experiments/predicts/vgg_SCNN_DULR_w9/driver_37_30frame/05181432_0203.MP4/'
imgNameEst1 = '02700_1_avg.png'
imgNameEst2 = '02700_2_avg.png'
imgNameEst3 = '02700_3_avg.png'
imgNameEst4 = '02700_4_avg.png'
name = '02700.png'

#load true info (.txt file)
file = open(imgDir + imgFile,'r')
fileContents = file.read()

# split content into each line segment, then parse into list
split = fileContents.splitlines()
# print split[2]
splitL1 = split[0].split(' ')
splitL2 = split[1].split(' ')
splitL3 = split[2].split(' ')
splitL4 = split[3].split(' ')

# print(len(splitL1))
# print(len(splitL2))
# print(len(splitL3))
# print(len(splitL4))

# print splitL4

# print splitL3

# get x and y for each line
x1 = splitL1[0:len(splitL1) - 1:2]
y1 = splitL1[1:len(splitL1) - 1:2]

x2 = splitL2[0:len(splitL2) - 1:2]
y2 = splitL2[1:len(splitL2) - 1:2]

x3 = splitL3[0:len(splitL3) - 1:2]
y3 = splitL3[1:len(splitL3) - 1 :2]

x4 = splitL4[0:len(splitL4) - 1:2]
y4 = splitL4[1:len(splitL4) - 1:2]

#convert to float values
x1 = list(map(float,x1))
y1 = list(map(float,y1))

x2 = list(map(float,x2))
y2 = list(map(float,y2))

x3 = list(map(float,x3))
y3 = list(map(float,y3))

x4 = list(map(float,x4))
y4 = list(map(float,y4))

# scale each values representing from truth
x1 = [x / 2.05 for x in x1]
y1 = [x / 2.05 for x in y1]

x2 = [x / 2.05 for x in x2]
y2 = [x / 2.05 for x in y2]

x3 = [x / 2.05 for x in x3]
y3 = [x / 2.05 for x in y3]

x4 = [x / 2.05 for x in x4]
y4 = [x / 2.05 for x in y4]


# Load 4 lane images and combine together into a new image
# place on image, then cycle through each image and overlay only nonzeroes
# im = plt.imread(imgDir + imgName)
im1 = plt.imread(imgDirEst + imgNameEst1)
im2 = plt.imread(imgDirEst + imgNameEst2)
im3 = plt.imread(imgDirEst + imgNameEst3)
im4 = plt.imread(imgDirEst + imgNameEst4)

im = im1 + im2 + im3 + im4

fig = plt.figure()
# plot image
implot = plt.imshow(im)

# put info on
plt.scatter(x = x1, y = y1, c = 'r', s = 40)
plt.scatter(x = x2, y = y2, c = 'g', s = 40)
plt.scatter(x = x3, y = y3, c = 'b', s = 40)
plt.scatter(x = x4, y = y4, c = 'y', s = 40)

# show image
# plt.show()

plt.savefig(name)
# plot the first image