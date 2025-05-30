import cv2
import mediapipe as mp
import numpy as np

# Teach the computer to see hands! 🤲
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# List of gestures we want to detect✌️❤️🔺
gestures = {
    0: "Peace ✌️",
    1: "Heart ❤️",
    2: "Triangle 🔺",
    3: "Circle ⭕",
    4: "Rectangle ▭"
}

# Open the webcam! 📸
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip the image (so it's not mirror-like)
    frame = cv2.flip(frame, 1)
    
    # Let the computer see hands in the frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand points (like connecting dots!)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Simple trick to detect gestures (for kids!)
            fingers_up = 0
            if landmarks[8*3+1] < landmarks[6*3+1]:  # Index finger
                fingers_up += 1
            if landmarks[12*3+1] < landmarks[10*3+1]:  # Middle finger
                fingers_up += 1
            if landmarks[16*3+1] < landmarks[14*3+1]:  # Ring finger
                fingers_up += 1
            if landmarks[20*3+1] < landmarks[18*3+1]:  # Pinky finger
                fingers_up += 1
            
            # Guess the gesture based on fingers
            if fingers_up == 2 and landmarks[4*3] > landmarks[2*3]:  # Peace ✌️
                gesture = 0
            elif fingers_up == 2 and landmarks[4*3] < landmarks[2*3]:  # Heart ❤️
                gesture = 1
            elif fingers_up == 3:  # Triangle 🔺
                gesture = 2
            elif landmarks[4*3+1] > landmarks[2*3+1] and abs(landmarks[8*3] - landmarks[4*3]) < 0.1:  # Circle ⭕
                gesture = 3
            else:  # Rectangle ▭
                gesture = 4
            
            # Show the gesture name on screen! 🎉
            cv2.putText(frame, gestures[gesture], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the camera feed
    cv2.imshow("Hand Gesture Detector", frame)
    
    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()