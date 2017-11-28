import cv2
import numpy as np

# 学習器(cascade.xml)の指定
Cascade = cv2.CascadeClassifier('./model_HOG/cascade.xml')
# 予測対象の画像の指定
img = cv2.imread('/vagrant/WIN_20171128_15_42_59_Pro.jpg')
img = cv2.resize(img, (320, 160))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
point = Cascade.detectMultiScale(gray, 1.06, 4, 0, (50, 50), (80, 80))

if len(point) > 0:
  for rect in point:
    cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=2)
else:
  print("no detect")

cv2.imwrite('detected.jpg', img)
