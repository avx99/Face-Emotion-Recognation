import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh =  mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        
    def findFaceMesh(self, img, draw=True):
         # Convert from BRG to RGB
         self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         self.result = self.faceMesh.process(self.imgRGB)
         faces = []
         if self.result.multi_face_landmarks:
             if draw:
                 for faceLms in self.result.multi_face_landmarks:
                     self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                              self.drawSpec,self.drawSpec)
                     
                     face = []
                     for id, lm in enumerate(faceLms.landmark):
                         ih, iw, ic = img.shape
                         x, y = int(lm.x*iw), int(lm.y*ih)
                         face.append([x,y])
            
                    
                     faces.append(face)
         return img, faces
        
    
def main():
    cap = cv2.VideoCapture("videos/2.mp4")
    
    p_time = 0
    detector = FaceMeshDetector(maxFaces=4)
    while True:
        success, img = cap.read()
        if(success):
            img, faces = detector.findFaceMesh(img, True)
            # Calculate num of frame per second
            c_time = time.time()
            fps = 1 / (c_time-p_time)
            p_time = time.time()
            
            # Print frames
            cv2.putText(img,f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255,0,0),3)
            cv2.imshow("Image",img)
            cv2.waitKey(1)
        else:
            break
        
def main2():
    image = cv2.imread("images/test1.jpg")
    
    detector = FaceMeshDetector(maxFaces=4)

    marked_image, faces = detector.findFaceMesh(image, True)
    
    cv2.imshow("Marked Image",marked_image)
    

if __name__ == "__main__":
    main2()    


