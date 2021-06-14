# Face Detection Using OpenCV
import cv2

# Load an image
img = cv2.imread('./images/test1.jpg')

# load the pre-trained model detector
classifier = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

# perform face detection 
faces = classifier.detectMultiScale(img)

# print boundring boxes for each face detection
for face in faces:
    # extract
    x, y, w, h = face
    x2, y2 = x + w, y + h
    
    # draw a rectangle over the face
    cv2.rectangle(img, (x,y), (x2, y2), (0,0,255),1)
    
# show the images
cv2.imshow('face detection using opencv', img)
 
# keep the window open until we press a key
cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()

