import cv2
from cv2.data import haarcascades

face_cascade = cv2.CascadeClassifier('./recdata/haarcascade_frontalface_default.xml')
img = cv2.imread('./img/smilings.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

rate = 10

for face in faces:
    fx, fy, fw, fh = face
    roi = img[fy:fy + fh, fx:fx + fw]
    roi = cv2.resize(roi, (fw // rate, fh // rate))
    roi = cv2.resize(roi, (fw, fh))
    img[fy:fy + fh, fx:fx + fw] = roi
    cv2.imshow('mosaic', img)

cv2.waitKey()
cv2.destroyAllWindows()