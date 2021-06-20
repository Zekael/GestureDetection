import mediapipe as mp
import cv2


class HandDetection:
    """class for detecting and drawing hands on images"""

    def __init__(self,static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands=max_num_hands
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands=mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_tracking_confidence=self.min_tracking_confidence,min_detection_confidence=self.min_detection_confidence,max_num_hands=self.max_num_hands,static_image_mode=self.static_image_mode)


    def detect(self,image,draw=True):

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if draw:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


        return image, results