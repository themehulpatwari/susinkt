import cv2
import mediapipe as mp
import time
from collections import deque
import math


capture = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


palm_indices = [0, 5, 9, 13, 17]  # wrist, base of each finger


pointer_indices = [8, 12]  # index tip, middle tip

positions_queue = deque(maxlen=30)

# Gesture cooldown
gesture_cooldown = 0.8
last_gesture_time = 0

# Time-decay factor
tau = 0.2

# Minimum gesture velocity (relative units per second)
min_gesture_velocity = 0.07


while True:
    success, frame = capture.read()
    if not success:
        break
    
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)
    
    pointer_relative_x = None
    pointer_relative_y = None
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate palm center (average of palm landmarks)
            palm_x = sum(hand_landmarks.landmark[i].x for i in palm_indices) / len(palm_indices)
            palm_y = sum(hand_landmarks.landmark[i].y for i in palm_indices) / len(palm_indices)
            
            # Calculate average pointer position (index + middle fingertips)
            pointer_x = sum(hand_landmarks.landmark[i].x for i in pointer_indices) / len(pointer_indices)
            pointer_y = sum(hand_landmarks.landmark[i].y for i in pointer_indices) / len(pointer_indices)
            
            # Key: Track ONLY the pointer position relative to palm
            # This automatically ignores whole-hand movement!
            pointer_relative_x = pointer_x - palm_x
            pointer_relative_y = pointer_y - palm_y
            
            # Draw visualization
            palm_px = int(palm_x * frame.shape[1])
            palm_py = int(palm_y * frame.shape[0])
            pointer_px = int(pointer_x * frame.shape[1])
            pointer_py = int(pointer_y * frame.shape[0])
            
            cv2.circle(frame, (palm_px, palm_py), 10, (0, 255, 0), -1)  # green palm
            cv2.circle(frame, (pointer_px, pointer_py), 10, (255, 0, 0), -1)  # blue pointers
            cv2.line(frame, (palm_px, palm_py), (pointer_px, pointer_py), (255, 255, 0), 2)
    
    # Add to positions queue
    current_time = time.time()
    if pointer_relative_x is not None:
        positions_queue.append((pointer_relative_x, pointer_relative_y, current_time))
    
    # Calculate velocity using time-weighted smoothing
    if len(positions_queue) >= 5:  # need some history
        # Get current and smoothed past position
        current_x, current_y, _ = positions_queue[-1]
        
        # Time-weighted average of past positions
        x_sum = y_sum = w_sum = 0
        for px, py, t_stamp in list(positions_queue)[:-1]:  # exclude current
            weight = math.exp(-(current_time - t_stamp) / tau)
            x_sum += px * weight
            y_sum += py * weight
            w_sum += weight
        
        if w_sum > 0:
            avg_past_x = x_sum / w_sum
            avg_past_y = y_sum / w_sum
            
            # Calculate velocity (change in RELATIVE position)
            oldest_time = positions_queue[0][2]
            dt = current_time - oldest_time
            
            if dt > 0:
                dx = current_x - avg_past_x
                dy = current_y - avg_past_y
                
                vx = dx / dt
                vy = dy / dt
                
                # Gesture detection
                if current_time - last_gesture_time > gesture_cooldown:
                    if vy > min_gesture_velocity:
                        print("DOWN swipe detected")
                        last_gesture_time = current_time
                    elif vy < -min_gesture_velocity:
                        print("UP swipe detected")
                        last_gesture_time = current_time
                
                #print(f"Pointer velocity: vx={vx:.2f}, vy={vy:.2f}")
    
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()






