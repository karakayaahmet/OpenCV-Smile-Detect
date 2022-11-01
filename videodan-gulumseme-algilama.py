import cv2

cap = cv2.VideoCapture("smile.mp4")

face_cascade = cv2.CascadeClassifier("frontalface.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    if ret is False:
        break

    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_face, 1.3, 7)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    roi_face = frame[y:y+h, x:x+w]
    roi_gray = gray_face[y:y+h, x:x+w]

    smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 5)

    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_face, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()