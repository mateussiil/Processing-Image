from turtle import width
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_neighbors(matrix, row, column, radius=1):
  width, height = matrix.shape
  center = width / 2
  neighbors = []

  for j in range(column-1-radius, column+radius):
    for i in range(row-1-radius, row+radius):
      if  i >= 0 and i < len(matrix) and j >= 0 and j < len(matrix[0]):
        neighbors.append(matrix[i][j])
      else:
        neighbors.append(0)

  width = len(neighbors)

  center = int(width / 2)
  
  del neighbors[center]
  
  return neighbors

def main():
  img = cv2.imread('Algorithm/Averaging filter/noisysalterpepper.png')

  new_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  row, column, = new_gray.shape

  radius = 10

  for i in range(radius, row-radius):
    for j in range(radius, column-radius):

      neighbors = get_neighbors(gray, i+1, j+1, radius)

      average = np.sort(neighbors)[int(len(neighbors)/2)]

      new_gray[i][j] = average

  cv2.imshow('img 1', gray)
  cv2.imshow('img 2', new_gray)
  cv2.waitKey(0)

if __name__ == "__main__":
  main()





