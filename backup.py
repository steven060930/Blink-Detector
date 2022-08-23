import cv2
import sys
import numpy as np
import pandas as pd

#cam = cv2.VideoCapture(1)
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, 150)

faceCascade = cv2.CascadeClassifier("C:\\Users\\yunzh\\opencv-test\\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("C:\\Users\\yunzh\\opencv-test\\haarcascade_eye.xml")

output_window = np.zeros((480, 640, 3))
text_pos = 0



if not cam.isOpened():
    raise IOError("Cannot Open webcam")

while True:
    text_pos += 40

    chk,  _inp = cam.read()

    _inp = cv2.flip(_inp, 1)

    # switch the input image into gray scale
    gray = cv2.cvtColor(_inp, cv2.COLOR_BGR2GRAY)

    #detect human face
    """
    minNeighbors defines how many objects are detected near the current one before it declares the face found. 
    minSize gives the minimum size of each deteced object window.
    """
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    #print the number of faces detected to terminal
    print('Found', len(faces), 'face', sep=" ") if len(faces) == 1 else print('Found', len(faces), 'faces', sep=" ")

    for (x, y, w, h) in faces:
        cv2.rectangle(_inp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #eye detection
        eyes = eyeCascade.detectMultiScale(gray)

        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(_inp, (e_x, e_y), (e_x+e_w, e_y+e_h), (255, 0, 0), 2)



    #print(chk)
    cv2.imshow('Video', _inp)
    cv2.imshow('Console', output_window)

    _ch = cv2.waitKey(1)
    if _ch & 0xFF == ord('q'):
        break

cam.release()
    
cv2.destroyAllWindows()