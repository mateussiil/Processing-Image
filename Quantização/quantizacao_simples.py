import numpy as np
import cv2

img = cv2.imread('/content/sample_data/test2.png', 0)
q = 2

img = np.uint8(img/q)*q
cv2.imshow('image_trans', img)
