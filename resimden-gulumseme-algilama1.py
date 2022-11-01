import cv2

img = cv2.imread("smile.jpg")
smile_cascade = cv2.CascadeClassifier("smile.xml")
face_cascade = cv2.CascadeClassifier("frontal.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)



