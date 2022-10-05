import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
img = cv2.imread('GeometricTransformation/red_panda.jpg')

def translate(image, tx, ty):
  img_t = np.zeros_like(image)
  img_t[:-ty,:-tx] = img[ty:,tx:]

  return img_t

def scale(image, sw, sh):
  if(sw<0 or sw<0):
    raise Exception("Sorry, no numbers below zero")

  img_s = np.zeros_like(image)
  img_s = img[::sw,::sh]
  
  return img_s

def rotation(image, degree):
  img_r = np.zeros_like(image)
  rads = math.radians(degree)

  original_width, original_height, channel = image.shape

  midx,midy = (original_width//2, original_height//2)

  for i in range(original_width):
    for j in range(original_height):
      x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
      y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)

      x=round(x)+midx 
      y=round(y)+midy 

      if (x>=0 and y>=0 and x<image.shape[0] and  y<image.shape[1]):
        img_r[i,j,:] = image[x,y,:]

  return img_r 

def scale2(image, sw, sh):
  if(sw<0 or sw<0):
    raise Exception("Sorry, no numbers below zero")

  original_width, original_height, channel = image.shape
  img_t = np.zeros_like(image)

  for i in range(original_width - 1):
    for j in range(original_height - 1):
      x = (1 + int(i / sw))/original_width
      y = (1 + int(j / sh))/original_height

      img_t[i + 1, j + 1] = img[int(i / sw), int(j / sh)]
  
  return img_t


trans_img = translate(img, 100, 50)
scale_img = scale(img, 2, 2)
rot_img = rotation(img, 45)

cv2.imshow('img1 translate', trans_img)
cv2.imshow('img2 scale', scale_img)
cv2.imshow('img3 rotation', rot_img)

cv2.imshow('img4 rotation + scale', rotation(scale_img, 45))


cv2.waitKey(0)

# print(translate(img, 2, 1))
