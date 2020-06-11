import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

fullImage = cv2.imread("img.jpg", 0)
(nWidth, nHeight) = (128, 128)  # nxn


def histogram(title, segment):
    plt.plot(cv2.calcHist(segment, [0], None, [256], [0, 256]))
    plt.show()
    plt.title(title)


# Ex1: Sliding Window
# im; image n: (n*n) size s:stride
def slidingWindow(im, n, s):
    # slide a window across the image
    for y in range(0, im.shape[0], s):
        for x in range(0, im.shape[1], s):
            yield x, y, im[y:y + n[1], x:x + n[0]]


def slide(img, size=(nWidth, nHeight), stride=nWidth):
    # loop over the sliding window
    for (x, y, window) in slidingWindow(img, size, stride):
        if window.shape[0] < nWidth or window.shape[1] < nHeight:
            continue

        # Creates a new Image each iteration to prevent overlapping rectangles
        clone = fullImage.copy()
        cv2.rectangle(clone, (x, y), (x + nWidth, y + nHeight), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        # Keeps refreshing new image every 0.025s
        cv2.waitKey(1)
        time.sleep(0.15)


# slide(fullImage, (80, 80), 20)

# Ex2:  Convolution on RoI
sobelKernelx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
sobelKernely = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def sobelSector(image, roi):
    img = image[0:roi, 0:roi]
    G = np.zeros(shape=(img.shape[0], img.shape[1]))

    # Apply the Sobel operator
    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            Gx = np.sum(np.multiply(sobelKernelx, img[i:i + 3, j:j + 3]))  # x direction
            Gy = np.sum(np.multiply(sobelKernely, img[i:i + 3, j:j + 3]))  # y direction
            G[i + 1, j + 1] = np.sqrt(Gx ** 2 + Gy ** 2)  # calculate the magnitude

    # Display the original image and the Sobel filtered image
    fig2 = plt.figure(2)
    ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
    ax1.imshow(img)
    ax2.imshow(G, cmap=plt.get_cmap('gray'))
    fig2.show()


# roi = 300
# sobelSector(fullImage, roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def sobel(img):
    G = np.zeros(shape=(img.shape[0], img.shape[1]))

    # Apply the Sobel operator
    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            Gx = np.sum(np.multiply(sobelKernelx, img[i:i + 3, j:j + 3]))  # x direction
            Gy = np.sum(np.multiply(sobelKernely, img[i:i + 3, j:j + 3]))  # y direction
            G[i + 1, j + 1] = np.sqrt(Gx ** 2 + Gy ** 2)  # calculate the magnitude
    return G


# Ex3: Convolution on the whole image
def sobelFull(img):
    sobel_image = sobel(img)
    # Display the original image and the Sobel filtered image
    figure = plt.figure(2)
    original_plot, sobel_plot = figure.add_subplot(121), figure.add_subplot(122)
    original_plot.imshow(img)
    sobel_plot.imshow(sobel_image, cmap=plt.get_cmap('gray'))
    figure.show()

    plt.hist(img.ravel(), 256, [0, 256], label='Original Image')
    plt.hist(sobel_image.ravel(), 256, [0, 256], label='Sobel Image')
    plt.legend(loc='upper right')
    plt.show()


# sobelFull(fullImage)


def sobelFullWithVisibleWindow(img):
    for (x, y, window) in slidingWindow(img, n=(nWidth, nHeight), s=nWidth):
        print(window.shape)

        if window.shape[0] < nWidth or window.shape[1] < nHeight:
            continue

        g = sobel(window)

        # Creates a new Image each iteration to prevent overlapping rectangles
        clone = img.copy()
        cv2.rectangle(clone, (x, y), (x + nWidth, y + nHeight), (0, 255, 0), 2)
        cv2.imshow("OG", clone)
        cv2.imshow("Sobel", g)  # Issue with how OpenCV shows the sobel output
        # Keeps refreshing new image every few ms
        cv2.waitKey(1)
        time.sleep(2)


# Ex4: Bilinear and Gaussian kernels
def conv(img):
    for (x, y, window) in slidingWindow(fullImage, n=(nWidth, nHeight), s=nWidth):
        if window.shape[0] < nWidth or window.shape[1] < nHeight:
            continue
        print(window.shape)
        print(window)

        # kernel = cv2.GaussianBlur(img, (10, 10), cv2.BORDER_DEFAULT)
        # dst = cv2.filter2D(window, -1, kernel)
        # cv2.imshow("Gaussian", dst)

        # Works with OpenCV
        blur = cv2.GaussianBlur(window, (5, 5), 0)
        cv2.imshow("blur", blur)

        # Creates a new Image each iteration to prevent overlapping rectangles
        clone = fullImage.copy()
        cv2.rectangle(clone, (x, y), (x + nWidth, y + nHeight), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        # Keeps refreshing new image every 0.025s
        cv2.waitKey(0)
        time.sleep(0.15)


# Ex4: bilateral and Gaussian kernels
def biKernel(img):
    for (x, y, window) in slidingWindow(fullImage, n=(nWidth, nHeight), s=nWidth):
        if window.shape[0] < nWidth or window.shape[1] < nHeight:
            continue
        print(window.shape)
        print(window)

        clone = fullImage.copy()
        cv2.bilateralFilter(img, clone, 9, 75, 75, cv2.BORDER_DEFAULT)

        # Creates a new Image each iteration to prevent overlapping rectangles
        cv2.rectangle(clone, (x, y), (x + nWidth, y + nHeight), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        # Keeps refreshing new image every 0.025s
        cv2.waitKey(1)
        time.sleep(0.15)


def makeGaussian():
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))


# Ex1
print("Ex1")
# slide(fullImage, (128, 128), 40)

# Ex2
# roi = 300
# sobelSector(fullImage, roi)

# Ex3
# sobelFull(fullImage)

# Ex3 with Window
# sobelFullWithVisibleWindow(fullImage)

# Ex4
conv(fullImage)
# biKernel(fullImage)
