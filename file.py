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
        X,Y,X2,Y2 = util_functions.bounding_box(frame, result.multi_hand_landmarks)

        canvas_img = util_functions.cropped_image(frame, X, Y, X2, Y2)
        

    cv.imshow("Camera", frame)

    key = cv.waitKey(1)
    
    if(key == ord('q')):
        break

    if(key == ord('s')):
        dataset = 'B'
        util_functions.save_image(canvas_img, dataset)

vid.release()
cv.destroyAllWindows()