import cv2
from cv2.data import haarcascades

face_cascade = cv2.CascadeClassifier('./recdata/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./recdata/haarcascade_eye.xml')

img = cv2.imread('./img/children.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray)
print(faces)

for face in faces:
    fx, fy, fw, fh = face

    center = (fx + fw // 2, fy + fh // 2)
    radius = max(fw, fh) // 2

    cv2.circle(img, center, radius, (255, 0, 0), 2)

    eyes = eye_cascade.detectMultiScale(gray[fy:fy + fh, fx:fx + fw])
    print(eyes)

    for eye in eyes:
        ex, ey, ew, eh = eye
        cv2.rectangle(img, (ex + fx, ey + fy), (ex + fx + ew, ey + fy + eh), (0, 255, 0), 2)

cv2.imshow('Image with Circular Bounding Box', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

