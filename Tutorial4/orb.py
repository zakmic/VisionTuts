import cv2
from matplotlib import pyplot as plt


book = cv2.imread("3_colour.jpeg", 1)  # train image
no_book = cv2.imread('2_colour.jpeg', 1)  # query image

orb = cv2.ORB_create(nfeatures=1500)

keypoints_orb, descriptorsOrb = orb.detectAndCompute(book, None)

nobook_keypoints_orb, descriptorsOrbNo = orb.detectAndCompute(no_book, None)

bookOrb = cv2.drawKeypoints(book, keypoints_orb, None)

no_bookOrb = cv2.drawKeypoints(no_book, keypoints_orb, None)

cv2.imshow("ORB Book", bookOrb)

cv2.imshow("ORB no Book", no_bookOrb)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(book, None)
kp2, des2 = orb.detectAndCompute(no_book, None)


#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.match(des1, des2)

img3 = cv2.drawMatches(book, kp1, no_book, kp2, matches, flags=2, outImg=None)
plt.imshow(img3), plt.show()
cv2.imshow("Matching", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()