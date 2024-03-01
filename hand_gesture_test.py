import mediapipe as mp
import cv2
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/Users/klsharma22/Desktop/ASLFingerSpelling/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)
landmarker = HandLandmarker.create_from_options(options)
hand_landmarks = []
while cap.isOpened():
    time_ms = int(round(time.time() * 1000))
    ret, frame = cap.read()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    results = landmarker.detect_async(mp_image, time_ms)
    hand_landmarks.append(results)
    print(results)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(hand_landmarks[0])