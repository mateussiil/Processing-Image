import cv2
import numpy as np
from matplotlib import pyplot as plt


def kmeans(img, K):
  Z = img.reshape((-1,3))
  Z = np.float32(Z)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
  center = np.uint8(center)
  res = center[label.flatten()]
  return res.reshape((img.shape))


k = 2
img = cv2.imread('/content/sample_data/cagecinza.png')

resultado = kmeans(img, k)
cv2.imshow('image_trans', img)