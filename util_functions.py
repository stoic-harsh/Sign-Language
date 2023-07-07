import cv2 as cv
import mediapipe as mp


mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils


def bounding_box(frame, multi_hand_landmarks):
        
    # dimensions for bounding box
    h, w, _ = frame.shape
    x1=y1=w
    x2=y2=0
    offset = 25
    
    for hand_landmarks in multi_hand_landmarks:
        for id, landmark in enumerate(hand_landmarks.landmark):
            
            x1, y1 = min(x1, int(landmark.x*w)), min(y1, int(landmark.y*h))
            x2, y2 = max(x2, int(landmark.x*w)), max(y2, int(landmark.y*h))

        # bounding box
        cv.rectangle(frame, (x1-offset,y1-offset), (x2+offset,y2+offset), (0,128,0), 4)

        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # return frame
