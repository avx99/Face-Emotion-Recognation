import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

# def sketch(img):
#     imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     imgGrayBlur = cv2.GaussianBlur(imgGray,(5,5),0)
#     canny = cv2.Canny(imgGrayBlur,10,70)
#     ret ,mask = cv2.threshold(canny,70,255,cv2.THRESH_BINARY_INV)
#     return mask


# classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# eyesClassifier = cv2.CascadeClassifier("haarcascades/haarcascade_eye_tree_eyeglasses.xml")
# detector = MTCNN()

# cap = cv2.VideoCapture(0)
# detector = MTCNN()
# while True:
#     ret ,frame = cap.read()
#     # grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Live Sketch', frame)

#     if cv2.waitKey(1) == 13:
#         break
    
# cap.release()
# cv2.destroyAllWindows()



detector = MTCNN()

image = cv2.imread("img.jpg")
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print(result)


