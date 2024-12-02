import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img_file = "./img/children.jpg"
image = cv2.imread(img_file)

# backends = [
#     'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface',
#     'mediapipe', 'yolov8', 'yunet', 'centerface']

# dlib, fastmtcnn,
# mediapipe, yolov8
backends = [
    'opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet', 'centerface']
timing = []
images = []

for engine in backends:
    print(engine)
    image = cv2.imread(img_file)
    start = time.time()
    detections = DeepFace.extract_faces(img_path=img_file,
                                        detector_backend=engine,
                                        enforce_detection=False)
    # print(detections)
    end = time.time()
    timing.append((end-start)*1000)

    for face in detections:
        facial_area = face['facial_area']
        print(facial_area)
        cv2.rectangle(image, (facial_area['x'], facial_area['y']),
                      (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']),
                      (255, 0, 0), 2)
        cv2.putText(image, engine, (10, 20), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 205, 0), 2)
        images.append(image)

manyimages = cv2.vconcat(images)
plt.bar(backends, timing, color='skyblue', alpha=0.7)
plt.xlabel('Engines')
plt.ylabel('RunTime')
plt.show()

cv2.imshow('img', manyimages)
# cv2.imwrite('mergedimage.jpg', manyimages)
cv2.waitKey(0)
cv2.destroyAllWindows()
