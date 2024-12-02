from retinaface import RetinaFace
import cv2
import time
start = time.time()

imgfile = './img/graduate.jpg'

resp = RetinaFace.detect_faces(imgfile)
print(resp)

image = cv2.imread(imgfile)

for key in resp.keys():
    face = resp[key]
    farea = face['facial_area']
    score = face['score']

    cv2.rectangle(image, (farea[0], farea[1]), (farea[2], farea[3]), (0, 205, 0), 2)

    cv2.putText(image, f'{score:.2f}', (farea[0], farea[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 205, 0), 2)

cv2.imshow('img', image)
end = time.time()
print((end-start)*1000)
cv2.waitKey(0)
cv2.destroyAllWindows()