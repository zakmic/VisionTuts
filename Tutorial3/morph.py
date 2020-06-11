from _warnings import filters

import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(title, segment):
    plt.plot(cv2.calcHist(segment, [0], None, [256], [0, 256]))
    plt.show()
    plt.title(title)


# Import all images
coins = cv2.imread("Euro_Coins.jpg", 0)
shapes = cv2.imread("shapes.jpg", 0)
text = cv2.imread("text.png", 0)

# Apply treshold to image
rets, tresh = cv2.threshold(text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('binary', tresh)

# Apply connected components
ret, labels = cv2.connectedComponents(tresh)

# # # Show the connected components 1 by 1.
# for label in range(1, ret):
#     mask = np.array(labels, dtype=np.uint8)
#     mask[labels == label] = 255
#     cv2.imshow('component', mask)
#     cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8)

plt.hist(tresh.flatten(), bins=[-.5, .5, 1.5], ec="k")
plt.title("Default")
plt.xticks((0, 1))
plt.show()

# Dilate
dilation = cv2.dilate(tresh, kernel, iterations=2)
cv2.imshow('dilate', dilation)

plt.hist(dilation.flatten(), bins=[-.5, .5, 1.5], ec="k")
plt.title("Dilation 2")
plt.xticks((0, 1))
plt.show()

# Erosions
erosion = cv2.erode(tresh, kernel, iterations=1)

cv2.imshow('erode', erosion)

plt.hist(erosion.flatten(), bins=[-.5, .5, 1.5], ec="k")
plt.title("Erosion")
plt.xticks((0, 1))
plt.show()

# Closing
closing = cv2.dilate(tresh, kernel, iterations=1)
closing = cv2.erode(closing, kernel, iterations=1)

plt.hist(closing.flatten(), bins=[-.5, .5, 1.5], ec="k")
plt.title("Closing")
plt.xticks((0, 1))
plt.show()

cv2.imshow('closing', closing)

cv2.waitKey(0)
