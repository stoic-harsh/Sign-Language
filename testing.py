import cv2 as cv
import mediapipe as mp

import testing_util_functions

mpHands = mp.solutions.hands
detectHands = mpHands.Hands(max_num_hands=1)

vid = cv.VideoCapture(0)


while(True):
    _, frame = vid.read()

    frame_in_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = detectHands.process(frame_in_RGB)

    if(result.multi_hand_landmarks): 
        X,Y,X2,Y2 = testing_util_functions.bounding_box(frame, result.multi_hand_landmarks)

        testing_util_functions.classify_gesture(frame, X, Y, X2, Y2)


    cv.imshow("Camera", frame)

    key = cv.waitKey(1)
    
    if(key == ord('q')):
        break

vid.release()
cv.destroyAllWindows()