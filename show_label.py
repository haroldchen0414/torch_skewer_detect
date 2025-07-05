# -*- coding: utf-8 -*-
# author: haroldchen0414

import numpy as np
import imutils
import cv2
import os

# 标注为YOLO格式, 即[class_id, x_center, y_center, width, height]
imagePath = os.path.join("images", "train", "0.jpg")
labelPath = os.path.join("labels", "train", "0.txt")

image = cv2.imread(imagePath)
image = imutils.resize(image, width=720)
height, width = image.shape[:2]
labels = []

with open(labelPath, "r") as f:
    for line in f:
        info = line.strip().split(" ")
        classId, xCenter, yCenter, w, h = map(float, info[:])
        
        # 将归一化坐标转化为实际像素坐标
        centerX = int(xCenter * width)
        centerY = int(yCenter * height)

        labels.append((centerY, centerX, classId, xCenter, yCenter, w, h))

# 对标注信息进行排序, 首先按y坐标排序(从上到下), 若y坐标相同则按x坐标排序(从左到右)
labels.sort(key=lambda x: (x[0], x[1]))
count = len(labels)

for (i, (_, _, classId, xCenter, yCenter, w, h)) in enumerate(labels):
    boxW = int(w * width)
    boxH = int(h * height)

    # 计算矩形框左上角和右下角坐标
    x1 = int(xCenter * width - boxW / 2)
    y1 = int(yCenter * height - boxH / 2)
    x2 = int(xCenter * width + boxW / 2)
    y2 = int(yCenter * height + boxH / 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f"{i}", (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.putText(image, f"Total: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("annotated_image.jpg", image)