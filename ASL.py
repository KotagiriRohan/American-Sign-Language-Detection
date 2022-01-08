from typing_extensions import Annotated
import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
with mp_hands.Hands(static_image_mode=True) as hands:
    img = cv2.imread("images.jpg")
    img = cv2.flip(img,1)
    img_height,img_width,_ = img.shape
    annotated_image = img.copy()
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    cv2.imshow("handlandmark",annotated_image)
    cv2.waitKey(0)