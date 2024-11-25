# 2024.11.25 이미지처리수업
# Canny edge detector를 이용한 라인과 원의 검출

import cv2
import numpy as np

img = cv2.imread('./img/sudoku.png')
# 컬러이미지는 흑백변환
# img = cv2.imread("./img/children.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
print(lines)
# 4. 검출된 직선 그리기
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 직선 방정식으로부터 직선의 시작점과 끝점 계산
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # 직선 그리기
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()