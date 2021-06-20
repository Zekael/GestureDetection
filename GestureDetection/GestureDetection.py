import cv2
from HandTracking import HandDetection


captureDevice = cv2.VideoCapture(0)
detect = HandDetection(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)

while True:

    success,img=captureDevice.read()

    img,results=detect.detect(img,draw=True)

    cv2.imshow('MediaPipe Hands', img)
    if cv2.waitKey(1) & 0xFF == 27:
      break




captureDevice.release()