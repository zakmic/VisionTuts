import cv2
import numpy as np
from matplotlib import pyplot as plt

# Tutorial 1 Point Processing
# Exercise 1

fullImage = cv2.imread("img.jpg", 0)

sector1 = fullImage[0:240, 0:240]
sector2 = fullImage[240:480, 0:240]
sector3 = fullImage[240:480, 240:480]
sector4 = fullImage[0:240, 240:480]

SectoredImage = [sector1, sector2, sector3, sector4]


def histogramCV(title, segment):
    plt.plot(cv2.calcHist(segment, [0], None, [256], [0, 256]))
    plt.title(title)
    plt.show()


def histogram(title, segment):
    plt.hist(segment.ravel(), 256, [0, 256])
    plt.title(title)
    plt.show()


def isTreshold(img, treshvalue):
    return np.mean(img) > treshvalue


def treshold(img, treshvalue):
    return 1.0 * (img > treshvalue)


def powerTransform(segment, gamma):
    return np.array(255 * (segment / 255) ** gamma, dtype='uint8')


def bitplaneSlice(img):
    # Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # lst.append(np.binary_repr(img[i][j], width=8))  # width = no. of bits
            lst.append('{0:08b}'.format(img[i][j]))  # width = no. of bits

    # We have a list of strings where each string represents binary pixel value.
    # To extract bit planes we need to iterate over the strings and store the characters
    # corresponding to bit planes into lists.
    # Multiply with 2^(n-1) and reshape to reconstruct the bit image.
    eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
    seven_bit_img = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
    six_bit_img = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
    five_bit_img = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
    four_bit_img = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
    three_bit_img = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
    two_bit_img = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
    one_bit_img = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])

    # Concatenate these images for ease of display using cv2.hconcat() and Vertically concatenate
    return cv2.vconcat([cv2.hconcat([eight_bit_img, seven_bit_img, six_bit_img, five_bit_img]),
                        cv2.hconcat([four_bit_img, three_bit_img, two_bit_img, one_bit_img])])


# Original Images in Greyscale
cv2.imshow("Sector 1", sector1)
# cv2.imshow("Sector 2", sector2)
# cv2.imshow("Sector 3", sector3)
# cv2.imshow("Sector 4", sector3)


# Calculating Histogram for each segment
histogram("Segment 1 Default", sector1)
# histogram("Histogram segment 2", sector2)
histogram("Segment 3 Default", sector3)
# histogram("Histogram segment 4", sector4)

# Thresholding
treshValue = 127
tresh1 = treshold(sector1, treshValue)
# tresh2 = treshold(sector2, treshValue)
# tresh3 = treshold(sector3, treshValue)
# tresh4 = treshold(sector4, treshValue)

cv2.imshow("Tresholding 1", tresh1)
# cv2.imshow("Tresholding 2", tresh2)
# cv2.imshow("Tresholding 3", tresh3)
# cv2.imshow("Tresholding 4", tresh4)
# Checking if sector is light or dark

print("Sector 1: %s" % ("Light" if isTreshold(sector1, treshValue) else "Dark"))
print("Sector 2: %s" % ("Light" if isTreshold(sector2, treshValue) else "Dark"))
print("Sector 3: %s" % ("Light" if isTreshold(sector3, treshValue) else "Dark"))
print("Sector 4: %s" % ("Light" if isTreshold(sector4, treshValue) else "Dark"))

# If Image is Dark set lighter Gamma & Vice-Versa
powerSector1 = powerTransform(sector1, 2.2 if isTreshold(sector1, treshValue) else 0.4)
# powerSector2 = powerTransform(sector2, 1.5 if isTreshold(sector2, treshValue) else 3)
powerSector3 = powerTransform(sector3, 1.5 if isTreshold(sector3, treshValue) else 3)
# powerSector4 = powerTransform(sector4, 1.5 if isTreshold(sector4, treshValue) else 3)

cv2.imshow("Power Transform sector1", powerSector1)
# cv2.imshow("Power Transform sector2", powerSector2)
cv2.imshow("Power Transform sector3", powerSector3)
# cv2.imshow("Power Transform sector4", powerSector4)

# Compare Powers Segment's Histograms
histogram("Power Transform 1", powerSector1)
# histogram("Power Sector2", powerSector2)
histogram("Power Sector3", powerSector3)
# histogram("Power Sector4", powerSector4)

# Bitwise Splicing
bitplane = bitplaneSlice(sector1)
bitplane2 = bitplaneSlice(sector2)
bitplane3 = bitplaneSlice(sector3)
bitplane4 = bitplaneSlice(sector4)

cv2.imshow("Sector1 Bitplane ", bitplane)
# cv2.imshow("Sector2 Bitplane ", bitplane2)
# cv2.imshow("Sector3 Bitplane ", bitplane3)
# cv2.imshow("Sector4 Bitplane ", bitplane4)

cv2.waitKey(0)
cv2.destroyAllWindows()
