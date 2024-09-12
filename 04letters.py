import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
img = np.full((500, 500, 3), 255, dtype=np.uint8)

# sans-serif small
cv2.putText(img, "Plain", (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
# sans-serif normal
cv2.putText(img, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
# sans-serif bold
cv2.putText(img, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
# sans-serif normal X 2
cv2.putText(img, "Simplex  X 2", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))

# serif small
cv2.putText(img, "Complex Small", (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
# serif normal
cv2.putText(img, "Complex", (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
# serif bold
cv2.putText(img, "Complex", (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
# serif normal X 2
cv2.putText(img, "Complex", (200, 260), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255))

# OpenCV 이미지를 PIL 이미지로 변환
img_pil = Image.fromarray(img)

# 드로우 객체 생성
draw = ImageDraw.Draw(img_pil)

# 사용할 폰트 설정 (예: 나눔고딕, 크기 40)
font = ImageFont.truetype("./fonts/NanumGothic.ttf", 20)

# 텍스트를 이미지에 그림 (한글 포함)
draw.text((50, 470), "아름다운 강산 - 전영준", font=font, fill=(0, 0, 0))

# PIL 이미지를 다시 OpenCV 이미지로 변환
img = np.array(img_pil)

cv2.imshow('draw text', img)
cv2.waitKey(0)
cv2.destroyAllWindows()