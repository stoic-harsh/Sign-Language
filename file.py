import cv2 as cv

vid = cv.VideoCapture(0)

while(True):
    _, frame = vid.read()

    cv.imshow("Camera", frame)

    if(cv.waitKey(1) == ord('q')):
        break

vid.release()
cv.destroyAllWindows()