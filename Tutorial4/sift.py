import cv2
from matplotlib import pyplot as plt

book = cv2.imread("3_colour.jpeg", 1)  # train image
no_book = cv2.imread('2_colour.jpeg', 1)  # query image

sift = cv2.xfeatures2d.SIFT_create()

keypoints_sift, descriptorsSift = sift.detectAndCompute(book, None)
nobook_keypoints_sift, descriptorsSiftNo = sift.detectAndCompute(no_book, None)

bookSift = cv2.drawKeypoints(book, keypoints_sift, None)
no_bookSift = cv2.drawKeypoints(no_book, keypoints_sift, None)

cv2.imshow("Sift Book", bookSift)
cv2.imshow("Sift No Book", no_bookSift)

# find the keypoints and descriptors with Sift
kp1, des1 = sift.detectAndCompute(book, None)
kp2, des2 = sift.detectAndCompute(no_book, None)

bf = cv2.BFMatcher()
matches = bf.match(des1, des2, k=2)

img3 = cv2.drawMatches(book, kp1, no_book, kp2, matches, flags=2, outImg=None)
plt.imshow(img3), plt.show()

cv2.imshow("Matching", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
