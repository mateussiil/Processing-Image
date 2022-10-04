# Baseado em https://www.imageeprocessing.com/2017/12/k-means-clustering-on-rgb-image.html
# https://morioh.com/p/b6763f7527d5
# https://medium.com/nerd-for-tech/k-means-python-implementation-from-scratch-8400f30b8e5c

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def kmeans(X, k):
  diff = 1
  cluster = np.zeros(X.shape[0])
  centroids = X[np.random.choice(X.shape[0], size=k), :]

  while diff:
    for i, row in enumerate(X):
        mn_dist = float('inf')
        for idx, centroid in enumerate(centroids):
            d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
            # store closest centroid
            if mn_dist > d:
              mn_dist = d
              cluster[i] = idx
    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
    
    if np.count_nonzero(centroids-new_centroids) == 0:
        diff = 0
    else:
        centroids = new_centroids
  return centroids, cluster

img = cv2.imread('/content/sample_data/cagecinza.png', cv2.IMREAD_UNCHANGED)

# img_q = Kmeans(img, 10)
img_q = kmeans(img, 5)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('image_trans', img)