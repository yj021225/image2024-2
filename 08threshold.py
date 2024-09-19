# 2024.9.19
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 첫번째 방법
# 125로 중간 숫자를 고정
img = cv2.imread('./img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)
thresh_np = np.zeros_like(img)
thresh_np[img > 127] = 255
_, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("thr", thresh_np)
cv2.imshow("IMG", img)
cv2.imshow("cv2", thresh_cv)
cv2.waitKey()
cv2.destroyAllWindows()

# 두번째 방법
# 이미지에 따라 숫자를 조정
img = cv2.imread('./img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE)
_, t80 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
_, t100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
_, t120 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
_, t140 = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)

cv2.imshow("t80", t80)
cv2.imshow("t100", t100)
cv2.imshow("t120", t120)
cv2.imshow("t140", t140)
cv2.waitKey()
cv2.destroyAllWindows()

# 세번째 방법
# otsu 알고리즘 적용
_, t130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
t, totsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(t)
cv2.imshow("t130", t130)
cv2.imshow("totsu", totsu)
cv2.waitKey()
cv2.destroyAllWindows()

# 네번째 방법
# 적응형 문턱값 적용 : 주위 값에 따라 달라짐, blk_size = 9
img = cv2.imread('./img/sudoku.png', cv2.IMREAD_GRAYSCALE)
blk_size = 9
C = 5

ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(ret)

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, blk_size, C)

cv2.imshow("img", img)
cv2.imshow("totus", th1)
cv2.imshow("tmean", th2)
cv2.imshow("tgaussian", th3)

cv2.waitKey()
cv2.destroyAllWindows()