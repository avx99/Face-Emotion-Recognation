import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture("Videos/5.mp4")

p_time = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh =  mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    
    if(success):
        # Convert from BRG to RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = faceMesh.process(imgRGB)
    
        # Calculate num of frame per second
        c_time = time.time()
        fps = 1 / (c_time-p_time)
        p_time = time.time()
        
        if result.multi_face_landmarks:
            for faceLms in result.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                              drawSpec,drawSpec)
        
        # Print frames
        cv2.putText(img,f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0,255,0),3)
        cv2.imshow("Image",img)
    else:
        break
        
    cv2.waitKey(1)