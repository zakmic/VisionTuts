import cv2
import numpy as np

book = cv2.imread("3_colour.jpeg")
bookCopy = cv2.imread("3_colour.jpeg")


def cornerHarris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


def shi_tomasi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 5)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    return img


cv2.imshow('Corner Harris', cornerHarris(book))
cv2.imshow('Shi_Tomasi', shi_tomasi(bookCopy))

cv2.waitKey(0)
