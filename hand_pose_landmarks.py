import mediapipe as mp
import cv2

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(min_detection_confidence= 0.8, min_tracking_confidence= 0.5, max_num_hands= 2)
pose = mp_pose.Pose(min_detection_confidence= 0.8, min_tracking_confidence= 0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    hand_landmarks = hands.process(image)
    pose_landmarks = pose.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if hand_landmarks.multi_hand_landmarks and pose_landmarks.pose_landmarks:
        for num, hand in enumerate(hand_landmarks.multi_hand_landmarks):
            mp_drawings.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            mp_drawings.draw_landmarks(image, pose_landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Landmarks Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()