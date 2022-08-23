from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import dlib
import argparse
import imutils
import math

"""
steps to achieve blink detection:

1. open camera and video capture
2. converting the input to the gray scale
3. face detection
4. get landmarks for eyes
5. calculate the blink ratio 
6. prevent duplicating and repeat counting

"""
#cam = cv2.VideoCapture(1)
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW) #speed faster when using a webcam
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, 150)

faceCascade = cv2.CascadeClassifier("C:\\Users\\yunzh\\opencv-test\\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("C:\\Users\\yunzh\\opencv-test\\haarcascade_eye.xml")

output_window = np.zeros((480, 640, 3))
text_pos = 0


#function for calculating the eye aspect ratio 
def eye_aspect_ratio(eye):
    # vertical distance
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    # compute the ration by the following formula
    ratio = (A+B) / (2. * C)

    return ratio


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]
BLINK_RATIO_THRESHOLD = 0.094
EYE_AR_CONSEC_FRAMES = 3
prev_blink_time = -1
prev_mils = -1
TOTAL = 1
FPS = 18



if not cam.isOpened():
    raise IOError("Cannot Open webcam")
    #sys.exit(0)

while True:

    chk,  _inp = cam.read()

    _inp = cv2.flip(_inp, 1)

    # convert the input image to gray scale
    gray = cv2.cvtColor(_inp, cv2.COLOR_BGR2GRAY)

    #detect human face
    """
    minNeighbors defines how many objects are detected near the current one before it declares the face found. 
    minSize gives the size of each window.
    """
    rects = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=(60, 60),
        flags = cv2.CASCADE_SCALE_IMAGE
    )    


    faces, _, _  = detector.run(image = _inp, upsample_num_times = 0, adjust_threshold = 0.0)

    for (x, y, w, h) in rects:
        # draw the frame for detected faces
        cv2.rectangle(_inp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #eye detection
        eyes = eyeCascade.detectMultiScale(
            gray,
            minSize = (50, 50)
        )

        # draw the frame for detected eyes
        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(_inp, (e_x, e_y), (e_x+e_w, e_y+e_h), (255, 0, 0), 2)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_ratio = eye_aspect_ratio(landmarks[36:42])
        right_eye_ratio = eye_aspect_ratio(landmarks[42:48])
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        #prevent duplicate counting for blinks
        time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
        mils = int(time_now[0:2])*360000 + int(time_now[2:4])*6000 + int(time_now[4:6])*100 + int(time_now[6:8])
        print(mils-prev_mils)


        if blink_ratio < BLINK_RATIO_THRESHOLD:            

            time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
            mils = int(time_now[0:2])*360000 + int(time_now[2:4])*6000 + int(time_now[4:6])*100 + int(time_now[6:8])
            print(mils-prev_mils)


            if mils - prev_mils > 25 or prev_mils == -1:
                text_pos += 40
                #print(TOTAL)
                TOTAL += 1
                cv2.putText(output_window, "blink at {}".format(datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]), (10, text_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120))

                prev_blink_time = datetime.datetime.now().strftime("%H%M%S%f")
                prev_mils = int(prev_blink_time[0:2])*360000 + int(prev_blink_time[2:4])*6000 + int(prev_blink_time[4:6])*100 + int(prev_blink_time[6:8])


    #print(chk)
    cv2.imshow('Video', _inp)
    cv2.imshow('Console', output_window)

    #time.sleep(0.07)

    _ch = cv2.waitKey(1)
    if _ch & 0xFF == ord('q'): # use q to terminate the cv2 windows
        break

cam.release()
    
cv2.destroyAllWindows()