import cv2
from HandTracking import HandDetection
import keyboard
import csv
import time


"""
0=dummy
1=one
2=two
3=three
4=four
5=five
6=fist
7=devil

0=l
1=r

"""

captureDevice = cv2.VideoCapture(0)
detect = HandDetection(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.6)

gestureName = 0
handLR = 0

with(open(("./data/handData"+str(time.time())+".csv"),'w+',newline='')) as csvfile:
    while True:

        if keyboard.is_pressed('t'):
            gestureName = input("Gesture 0 dummy, 1 one, 2 peace, 3 okay, 4 halt, 5 five")
            handLR = input("l 0 or r 1: ")

        success,img=captureDevice.read()

        img,results=detect.detect(img,draw=True)


        if results.multi_hand_landmarks is not None:
            if keyboard.is_pressed(' '):
                row = [gestureName,handLR]

                for hand_landmark in results.multi_hand_landmarks:
                    for landmark in hand_landmark.landmark:
                        x = landmark.x
                        y = landmark.y
                        z = landmark.z

                        row.append(x)
                        row.append(y)
                        row.append(z)

                csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(row)
                print(row)


        cv2.imshow('Data Generation', img)

        #27 is esc
        if cv2.waitKey(1) & 0xFF == 27:
          break



csvfile.close()
captureDevice.release()