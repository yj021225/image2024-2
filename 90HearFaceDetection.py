import cv2
from cv2.data import haarcascades

face_cascade = cv2.CascadeClassifier('./recdata/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./recdata/haarcascade_eye.xml')
img = cv2.imread('./img/man_face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
print(faces)

for face in faces:
    fx, fy, fw, fh = face
    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
    eyes = eye_cascade.detectMultiScale(gray[fy:fy+fh, fx:fx+fw])
    print(eyes)
    for eye in eyes:
        ex, ey, ew, eh = eye
        cv2.rectangle(img, (ex + fx, ey + fy), (ex + fx + ew, ey + fy + eh), (0, 255, 0), 2)

cv2.imshow('gray', img)
cv2.waitKey()
cv2.destroyAllWindows()

