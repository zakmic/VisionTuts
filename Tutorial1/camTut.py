import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(title, segment):
    plt.plot(cv2.calcHist(segment, [0], None, [256], [0, 256]))
    plt.show()
    plt.title(title)


def isTreshold(img, treshvalue):
    return np.mean(img) > treshvalue


def powerTransform(segment, gamma):
    return np.array(255 * (segment / 255) ** gamma, dtype='uint8')


cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)
while (True):
    ret, frame = cap.read()

    # Segment the Frame
    sector1 = frame[0:240, 0:240]
    sector2 = frame[240:480, 0:240]
    sector3 = frame[240:480, 240:480]
    sector4 = frame[0:240, 240:480]

    # Convert Frames to Greyscale
    sector1 = cv2.cvtColor(sector1, cv2.COLOR_BGR2GRAY)
    sector2 = cv2.cvtColor(sector2, cv2.COLOR_BGR2GRAY)
    sector3 = cv2.cvtColor(sector3, cv2.COLOR_BGR2GRAY)
    sector4 = cv2.cvtColor(sector4, cv2.COLOR_BGR2GRAY)

    treshValue = 127
    powerSector1 = powerTransform(sector1, 2.2 if isTreshold(sector1, treshValue) else 0.4)
    powerSector2 = powerTransform(sector2, 2.2 if isTreshold(sector2, treshValue) else 0.4)
    powerSector3 = powerTransform(sector3, 2.2 if isTreshold(sector3, treshValue) else 0.4)
    powerSector4 = powerTransform(sector4, 2.2 if isTreshold(sector4, treshValue) else 0.4)

    cv2.imshow('frame1', powerSector1)
    cv2.imshow('frame2', powerSector2)
    cv2.imshow('frame3', powerSector3)
    cv2.imshow('frame4', powerSector4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
