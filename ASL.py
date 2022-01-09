import mediapipe as mp
import numpy as np
import cv2
from tensorflow import keras

model_new = keras.models.load_model('ASL_model.hdf5')
encoding = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'J': 10,
    'K': 11,
    'L': 12,
    'M': 13,
    'N': 14,
    'O': 15,
    'P': 16,
    'Q': 17,
    'R': 18,
    'S': 19,
    'T': 20,
    'U': 21,
    'V': 22,
    'W': 23,
    'X': 24,
    'Y': 25,
    'Z': 26,
}
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    letters = []
    vid = cv2.VideoCapture(0)
    count = 60
    letters.append("Output ")
    while(vid.isOpened()):

        _, img = vid.read()
        img = cv2.flip(img, 1)
        img_height, img_width, _ = img.shape
        test_np = np.zeros((img_height + 70, img_width,
                           img.shape[2]), dtype=np.uint8)
        test_np[:img_height, :img_width] = img.copy()
        annotated_image = img.copy()
        final_str = ""
        for i in letters:
            final_str = final_str+str(i)
        cv2.putText(annotated_image, final_str, (7, img_height),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (100, 255, 0), 1, cv2.LINE_4)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            cv2.imshow("handlandmark", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        if(count == 0):
            count = 60
            if not results.multi_hand_world_landmarks:
                cv2.imshow("handlandmark", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    count = 0
                    break
                continue
            keypoints = []
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                for data_point in hand_world_landmarks.landmark:
                    keypoints.append(data_point.x)
                    keypoints.append(data_point.y)
                    keypoints.append(data_point.z)
            y = model_new.predict(
                np.array(keypoints).astype('float64').reshape(-1, 63))
            y = y.tolist()
            num = y[0].index(max(y[0]))
            letter = list(encoding.keys())[
                list(encoding.values()).index(num+1)]
            letters.append(letter)
        cv2.imshow("handlandmark", annotated_image)
        count -= 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
