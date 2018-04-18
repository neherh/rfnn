import os, sys
import numpy as np
import cv2
from time import sleep

img_dir = '/home/vidavilane/Documents/repos/me640/pytorch/small_dataset/small_preds/driver_37_30frame/05181656_0251.MP4/00090.png'

img = cv2.imread(img_dir,0)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)


print(img[63])
winname = 'example'
# img2 = cv2.resize(img,(500,500))
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey()
cv2.destroyWindow(winname)