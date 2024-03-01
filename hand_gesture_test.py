import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence= 0.8, min_tracking_confidence= 0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # Detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
        image.flags.writeable = False # Setting flag to not render anythin on the image
        results = hands.process(image) # Detecting hand in the image

        image.flags.writeable = True # Setting the flag to write on the imgae
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(results)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()