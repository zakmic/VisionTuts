import cv2
from matplotlib import pyplot as plt

book = cv2.imread("3_colour.jpeg", 1)  # train image
no_book = cv2.imread('2_colour.jpeg', 1)  # query image

surf = cv2.xfeatures2d.SURF_create()

keypoints_surf, descriptorsSurf = surf.detectAndCompute(book, None)
nobook_keypoints_surf, descriptorsSurfNo = surf.detectAndCompute(no_book, None)

bookSurf = cv2.drawKeypoints(book, keypoints_surf, None)
no_bookSurf = cv2.drawKeypoints(no_book, keypoints_surf, None)

cv2.imshow("Surf", bookSurf)
cv2.imshow("Surf", no_bookSurf)

# find the keypoints and descriptors with ORB
kp1, des1 = surf.detectAndCompute(book, None)
kp2, des2 = surf.detectAndCompute(no_book, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
matches = bf.match(des1, des2)

img3 = cv2.drawMatches(book, kp1, no_book, kp2, matches[:10], flags=2, outImg=None)
plt.imshow(img3), plt.show()
cv2.imshow("Matching", img3)


cv2.waitKey(0)
cv2.destroyAllWindows()
