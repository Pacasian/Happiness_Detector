import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

cap = cv2.VideoCapture(0) 
def detect_face(img):
    face_img = img.copy()
    
    # Import from haar cascade ----required file face and smile file----
    face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_smile.xml')
    
    # Take the appropriate scale size ....
    face_rects = face_cascade.detectMultiScale(face_img,1.3, 5)
    
    # Takes starting piont and the ending point of the face
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,0,0), 5) 
        #Getting inside the face
        roi_gray = face_img[y:y + h, x:x + w]
        roi_img = face_img[y:y + h, x:x + w]
        
        #Detect the smile classifier
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2,minNeighbors=100,minSize=(25, 25))
        
        # Takes starting piont and the ending point of the smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_gray, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            
            #Checking the intensity of laugh
            sm_ratio = str(round(sw / sx, 3))
            s_ratio =sm_ratio
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(face_img, 'Smile meter : ' + sm_ratio, (10, 50), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
            if (float(s_ratio) > 1.0):
                cv2.putText(face_img, "Keep Smiling", (x,y-15), font, 1, (42, 244, 42), 2, cv2.LINE_AA)
            else:
                cv2.putText(face_img, "Sad", (x,y-5), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
    return face_img
while True: 
    ret, frame = cap.read(0) 
    # Pass the control to the function detect_face with an image as an parameter
    frame = detect_face(frame)
    
    cv2.imshow('Smile_Detection', frame) 
    c = cv2.waitKey(0) 
    if c == 27:
        break 

cap.release()
cv2.destroyAllWindows()
