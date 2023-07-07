import cv2 as cv
import mediapipe as mp

import util_functions

mpHands = mp.solutions.hands
detectHands = mpHands.Hands(max_num_hands=1)

vid = cv.VideoCapture(0)

while(True):
    _, frame = vid.read()

    frame_in_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = detectHands.process(frame_in_RGB)

    if(result.multi_hand_landmarks): 
        util_functions.bounding_box(frame, result.multi_hand_landmarks)
    
    cv.imshow("Camera", frame)

    if(cv.waitKey(1) == ord('q')):
        break

vid.release()
cv.destroyAllWindows()