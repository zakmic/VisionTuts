import cv2
import numpy as np
from matplotlib import pyplot as plt

# Tutorial 1 Point Processing
# Exercise 1

fullImage = cv2.imread("img.jpg", 1)

sector1 = fullImage[0:240, 0:240]
sector2 = fullImage[240:480, 0:240]
sector3 = fullImage[240:480, 240:480]
sector4 = fullImage[0:240, 240:480]

SectoredImage = [sector1, sector2, sector3, sector4]


def histogram(title, segment):
    plt.plot(cv2.calcHist(segment, [0], None, [256], [0, 256]))
    plt.show()
    plt.title(title)


def img_estim(img, thrshld):
    return np.mean(img) > thrshld


def powerTransform(segment, gamma):
    return np.array(255 * (segment / 255) ** gamma, dtype='uint8')


def bitplaneSlice(segment):
    out = []

    for k in range(0, 7):
        # create an image for each k bit plane
        plane = np.full((segment.shape[0], segment.shape[1]), 2 ** k, np.uint8)
        # execute bitwise and operation
        res = cv2.bitwise_and(plane, segment)
        # multiply ones (bit plane sliced) with 255 just for better visualization
        x = res * 255
        # append to the output list
        out.append(x)
    return np.hstack(out)


# Threshold
res, mask = cv2.threshold(sector1, 80, 255, cv2.THRESH_BINARY)
res2, mask2 = cv2.threshold(sector2, 80, 255, cv2.THRESH_BINARY)
res3, mask3 = cv2.threshold(sector3, 80, 255, cv2.THRESH_BINARY)
res4, mask4 = cv2.threshold(sector4, 80, 255, cv2.THRESH_BINARY)

# Checking if sector is light or dark
treshold = 127
print("Sector 1: %s" %("Light" if img_estim(mask, treshold) else "Dark"))
print("Sector 2: %s" %("Light" if img_estim(mask2, treshold) else "Dark"))
print("Sector 3: %s" %("Light" if img_estim(mask3, treshold) else "Dark"))
print("Sector 4: %s" %("Light" if img_estim(mask4, treshold) else "Dark"))

cv2.imshow("Treshold", mask)
cv2.imshow("Treshold2", mask2)
cv2.imshow("Treshold3", mask3)
cv2.imshow("Treshold4", mask4)

# Bitwise Not
s = cv2.bitwise_not(mask)
s2 = cv2.bitwise_not(mask2)
s3 = cv2.bitwise_not(mask3)
s4 = cv2.bitwise_not(mask4)

# Calculating Histogram for each segment
histogram("Histogram segment 1", sector1)
histogram("Histogram segment 2", sector2)
histogram("Histogram segment 3", sector3)
histogram("Histogram segment 4", sector4)

# Calculating Histogram for each segment with Binary Treshold
histogram("Treshold segment 1", mask)
histogram("Treshold segment 2", mask2)
histogram("Treshold segment 3", mask3)
histogram("Treshold segment 4", mask4)

# If Image is Dark set lighter Gamma & Vice-Versa
powerSector1 = powerTransform(sector1, 1.5 if img_estim(sector1, treshold) else 3)
powerSector2 = powerTransform(sector2, 1.5 if img_estim(sector2, treshold) else 3)
powerSector3 = powerTransform(sector3, 1.5 if img_estim(sector3, treshold) else 3)
powerSector4 = powerTransform(sector4, 1.5 if img_estim(sector4, treshold) else 3)

cv2.imshow("Power Transform sector1", powerSector1)
cv2.imshow("Power Transform sector2", powerSector2)
cv2.imshow("Power Transform sector3", powerSector3)
cv2.imshow("Power Transform sector4", powerSector4)

# Compare Powers Segment's Histograms
histogram("Power Sector1", powerSector1)
histogram("Power Sector2", powerSector2)
histogram("Power Sector3", powerSector3)
histogram("Power Sector4", powerSector4)

# Converting Image to Greyscale


# Bitwise Splicing
bitplane = bitplaneSlice(cv2.cvtColor(sector1, cv2.COLOR_BGR2GRAY))
bitplane2 = bitplaneSlice(cv2.cvtColor(sector2, cv2.COLOR_BGR2GRAY))
bitplane3 = bitplaneSlice(cv2.cvtColor(sector3, cv2.COLOR_BGR2GRAY))
bitplane4 = bitplaneSlice(cv2.cvtColor(sector4, cv2.COLOR_BGR2GRAY))

cv2.imshow("Sector1 Bitplane ", bitplane)
cv2.imshow("Sector2 Bitplane ", bitplane2)
cv2.imshow("Sector3 Bitplane ", bitplane3)
cv2.imshow("Sector4 Bitplane ", bitplane4)

# Compare Bitplane Segment's Histograms
histogram("Bitplane Sector 1", bitplane)
histogram("Bitplane Sector 2", bitplane2)
histogram("Bitplane Sector 3", bitplane3)
histogram("Bitplane Sector 4", bitplane4)

cv2.waitKey(0)
cv2.destroyAllWindows()
