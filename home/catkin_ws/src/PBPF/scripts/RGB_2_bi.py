import cv2
import numpy as np

img = cv2.imread('001.png')
cv2.imshow('Original Image', img)



gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_img)


t, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image', binary_img)
cv2.imwrite('0001.png', binary_img)
cv2.waitKey(0)
print("test")
