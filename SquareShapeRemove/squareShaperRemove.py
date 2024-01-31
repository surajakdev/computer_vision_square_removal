

import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_hands =mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
count_line_position = 650
# For static images:
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  
    # reading camera frame.
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # conversion bgr to rgb as mediapipe requires frames in bgr format to process image.
    results = hands.process(image)
    image_height, image_width, c = image.shape

    # orange line drawn.
    line = cv2.line(image, (0, 400), (700, 400), (255,127,0),3)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      # for loop for drawing landmarks on hands
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            # for connecting differnt points in hand and draw landmarks.
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()),
            
        print(f'Ring finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )

        array_of_tuples = np.array([(0, 400), (700, 400)])

        # x and y cordinate tuple from frames detected this tuple is of index fingers position.
        xcor = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        ycor= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height

        tuple_to_check = (xcor,ycor)
      
        for tuple_in_array in array_of_tuples:
          # for loop to check if the element with numerical value of coordinates detected tuple is within the numerical value range of line coordinates tuple.          
          if tuple_to_check[0] >= tuple_in_array[0] and tuple_to_check[1] <= tuple_in_array[1]:
            cv2.rectangle(image,(384,0),(510,128),(0,255,0),3)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('r'):
      break
cap.release()
cv2.destroyAllWindows()











