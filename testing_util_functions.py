import math
import cv2 as cv
import mediapipe as mp
import numpy as np

from keras.models import load_model

# Load the model
model = load_model("Model/keras_model.h5", compile=False)

# Load the labels
class_names = ['A', 'B']


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
        cv.rectangle(frame, (x1-offset,y1-offset), (x2+offset,y2+offset), (0,128,0), 2)

        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    return [x1-offset, y1-offset, x2+offset, y2+offset]



def classify_gesture(frame, X,Y,X2,Y2):
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


            # Resize the raw image into (224-height,224-width) pixels
            canvas = cv.resize(canvas, (224, 224), interpolation=cv.INTER_AREA)

            # Make the image a numpy array and reshape it to the models input shape.
            canvas = np.asarray(canvas, dtype=np.float32).reshape(1, 224, 224, 3)

            canvas = (canvas/127.5) -1

            # model predictions
            prediction = model.predict(canvas)
            index = np.argmax(prediction)

            confidence_score = str(np.round(prediction[0][index]*100))

            # displaying result
            cv.rectangle(frame, (X,Y-50), (X+50,Y-4), (255,255,255), -1)
            cv.putText(frame, class_names[index], (X+10, Y-10), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    