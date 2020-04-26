import cv2

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    sector1 = frame[0:240, 0:240]
    sector2 = frame[240:480, 0:240]
    sector3 = frame[240:480, 240:480]
    sector4 = frame[0:240, 240:480]

    res, mask = cv2.threshold(sector1, 80, 255, cv2.THRESH_BINARY)
    res2, mask2 = cv2.threshold(sector2, 80, 255, cv2.THRESH_BINARY)
    res3, mask3 = cv2.threshold(sector3, 80, 255, cv2.THRESH_BINARY)
    res4, mask4 = cv2.threshold(sector4, 80, 255, cv2.THRESH_BINARY)



cap.release()
cv2.destroyAllWindows()
