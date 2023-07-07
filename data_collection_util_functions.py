import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import time


mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

def bounding_box(frame, multi_hand_landmarks):
        
    # dimensions for bounding box
    h, w, _ = frame.shape
    x1=y1=w
    x2=y2=0
    offset = 25

    cv.circle(frame, (w,0), 6, (255,0,0), -1)
    
    for hand_landmarks in multi_hand_landmarks:
        for id, landmark in enumerate(hand_landmarks.landmark):
            
            x1, y1 = min(x1, int(landmark.x*w)), min(y1, int(landmark.y*h))
            x2, y2 = max(x2, int(landmark.x*w)), max(y2, int(landmark.y*h))

        # bounding box
        cv.rectangle(frame, (x1-offset,y1-offset), (x2+offset,y2+offset), (0,128,0), 4)

        # cv.putText(frame, "Thumb", (x1-offset,y1-offset-5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    return [x1-offset, y1-offset, x2+offset, y2+offset]



def cropped_image(frame, X,Y,X2,Y2):
    maxH, maxW, _ = frame.shape
    size = 200

    canvas = np.ones((size,size,3), np.uint8)*255

    if(X>0 and Y>0 and X2<maxW and Y2<maxH):
            
            cropped = frame[Y: Y2, X:X2]
            # cv.imshow("Crop", cropped)

            h, w = Y2-Y, X2-X
            aspectRatio = w/h

            if(aspectRatio < 1):
                calc_w = math.ceil(size*aspectRatio)
                resized_image = cv.resize(cropped, (calc_w, size))
                extra_gap = math.ceil((size-calc_w)/2)
                canvas[:, extra_gap:extra_gap+calc_w] = resized_image
            
            else:
                calc_h = math.ceil(size/aspectRatio)
                resized_image = cv.resize(cropped, (size, calc_h))
                extra_gap = math.ceil((size-calc_h)/2)
                canvas[extra_gap:extra_gap+calc_h, :] = resized_image

            
            cv.imshow("Canvas", canvas)
    return canvas


def save_image(img, location):
    cv.imwrite(f'Data/{location}/{time.time()}.jpg', img)