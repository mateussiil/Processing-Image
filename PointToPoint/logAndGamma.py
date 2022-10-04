import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow


img = cv2.imread('/Captura de tela 2022-06-22 150132.png', cv2.IMREAD_UNCHANGED)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gamma = 95
mag = 85


def calculate_gamma_correction(x, gamma):
  return ((x/255)**2)*gamma

def calculate_logarithm_operator(x, mag):

  log_value = math.log(1 + abs(mag))
  c = 255 / log_value

  return c*math.log1p(x)

def logarithm_operator(image, mag):
  for i in range(len(image)):
    for j in range(len(image[i])):
      [r,g,b] = img_rgb[i][j]
      image[i][j] = [calculate_logarithm_operator(r, gamma),calculate_logarithm_operator(g, gamma),calculate_logarithm_operator(b, gamma)]
  return image

def calculate_gamma_correction(image, gamma):
  for i in range(len(image)):
    for j in range(len(image[i])):
      [r,g,b] = image[i][j]
      image[i][j] = [calculate_gamma_correction(r, gamma),calculate_gamma_correction(g, gamma),calculate_gamma_correction(b, gamma)]
  return image

img_lo = logarithm_operator(img_rgb,mag)

cv2_imshow(img_lo)