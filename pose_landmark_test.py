import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence= 0.8, min_tracking_confidence= 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        resutls = pose.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if resutls.pose_landmarks:
                print(resutls.pose_landmarks)
            # for num, pos in enumerate(resutls.pose_world_landmarks):
                mp_drawing.draw_landmarks(image, resutls.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()

cv2.destroyAllWindows()
