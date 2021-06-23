import cv2
import numpy as np
from HandTracking import HandDetection
from GestureTrainer import ModelHandler


font = cv2.FONT_HERSHEY_SIMPLEX
captureDevice = cv2.VideoCapture(0)
detect = HandDetection(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.5)
model = ModelHandler()
model.loadModel("./Model/model20210623-085317")

while True:

    success,img=captureDevice.read()

    img,results=detect.detect(img,draw=True)

    if results.multi_hand_landmarks is not None:
        row = []  
        hand_landmark=results.multi_hand_landmarks[0]
        for landmark in hand_landmark.landmark:
            x = landmark.x
            y = landmark.y
            z = landmark.z

            row.append(x)
            row.append(y)
            row.append(z)

        row = np.array(row)
        row = row.reshape(1,63)
        #print(row.shape)
        pred = model.predict(row)
        predText = model.predictionToText(pred)
        print("Prediction: ",np.round(pred,3))
        cv2.putText(img,str(predText),(10,30),font, 1,(255,255,255,0),2)


    cv2.imshow('MediaPipe Hands', img)
    if cv2.waitKey(1) & 0xFF == 27:
      break




captureDevice.release()