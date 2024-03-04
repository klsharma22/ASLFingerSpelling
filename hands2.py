import cv2 
hand_cascade = cv2.CascadeClassifier('hand_cascade.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray , 1.1 , 2)
    for (x,y,w,h) in hands:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0 , 0 ,255), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.DestroyAllWindows()
