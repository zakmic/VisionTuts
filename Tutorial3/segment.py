import cv2
import numpy as np

text = cv2.imread("Capture.PNG", 1)
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((4, 4), np.uint8)
gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)

# closing = cv2.dilate(text, kernel2, iterations=1)

rets, tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('binary', tresh)
print(rets)

final = cv2.dilate(tresh, kernel, iterations=1)
# final = cv2.erode(closing, kernel, iterations=1)
cv2.imshow('final', final)

# Apply connected components
# ret, labels = cv2.connectedComponents(final)
#
# # # Show the connected components 1 by 1.
# for label in range(1, ret):
#     mask = np.array(labels, dtype=np.uint8)
#     mask[labels == label] = 255
#     cv2.imshow('component', mask)
#     cv2.waitKey(0)
cv2.waitKey(0)