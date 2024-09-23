# 2024.9.23 이미지분석 수업
# 이미지 연산
import cv2
import numpy as np

img1 = cv2.imread('./img/wing_wall.jpg')
img2 = cv2.imread('./img/yate.jpg')

img3 = img1 + img2
img4 = cv2.add(img1, img2)
img5 = (img1 / 2 + img2 / 2).astype(np.uint8)
print(img1)

cv2.imshow("wing", img1)
cv2.imshow("yate", img2)
cv2.imshow("add", img3)
cv2.imshow("cv2 add", img4)
cv2.imshow("/2 add", img5)
cv2.waitKey()
cv2.destroyAllWindows()

# 이미지 합성
# 각각 이미지의 비중은 alpha와 1-alpha
alpha = 0.5
blended = (img1 * alpha + img2 * (1-alpha)).astype(np.uint8)
cvblended = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0)

cv2.imshow("blended", blended)
cv2.imshow("cvblended", cvblended)
cv2.waitKey()
cv2.destroyAllWindows()

# alpha값 변경 트랙
win_name = "Alpha blending"
trackbar_name = 'fade'

def onChange(x):
    alpha = x / 100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    cv2.imshow(win_name, dst)

cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()