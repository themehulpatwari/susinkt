"""
Gesture controller for YouTube Shorts navigation.
Detects hand gestures (UP/DOWN swipes) and head pose (looking up/down) to control video scrolling.
Both detection methods share the same camera.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math
import threading
from queue import Queue


class HandGestureController:
    """
    Non-blocking gesture detector that runs in a separate thread.
    Detects both hand swipes and head pose for video navigation.
    Note: On macOS, video display must be disabled when running in background threads.
    """
    
    def __init__(self, show_video=False):
        self.capture = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_facemesh = mp.solutions.face_mesh
        self.hands = None
        self.face_mesh = None
        
        # Palm and pointer indices
        self.palm_indices = [0, 5, 9, 13, 17]  # wrist, base of each finger
        self.pointer_indices = [8, 12]  # index tip, middle tip
        
        # Position tracking
        self.positions_queue = deque(maxlen=30)
        
        # Gesture cooldown
        self.gesture_cooldown = 1.5  # Increased from 1.2 to 1.5s
        self.last_gesture_time = 0
        
        # Head pose tracking
        self.head_pose_active = False
        self.head_pose_start_time = 0
        self.head_pose_orientation = "Neutral"
        self.head_pose_timer = 0.5  # Reduced time for nod detection
        
        # Time-decay factor
        self.tau = 0.2
        
        # Minimum gesture velocity (relative units per second)
        self.min_gesture_velocity = 0.18  # Increased from 0.12 to 0.18 (much more deliberate swipe needed)
        
        # Thread control
        self.running = False
        self.thread = None
        self.gesture_queue = Queue()
        
        # Video display control
        self.show_video = show_video
        
        # For thread-safe video display
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        """Start the gesture detection in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.capture = cv2.VideoCapture(0)
        # Set camera to higher FPS if possible
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        # Reduce resolution for better performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize with optimized settings
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1  # Only track one hand for better performance
        )
        self.face_mesh = self.mp_facemesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            refine_landmarks=False  # Disable for better performance
        )
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the gesture detection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        
    def get_gesture(self):
        """
        Get the latest gesture if available.
        Returns: 'UP', 'DOWN', or None
        """
        if not self.gesture_queue.empty():
            return self.gesture_queue.get()
        return None
    
    def _detection_loop(self):
        """Main detection loop that runs in a separate thread."""
        while self.running:
            success, frame = self.capture.read()
            if not success:
                continue
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make RGB_frame non-writeable for better performance
            RGB_frame.flags.writeable = False
            
            # Process hand gestures
            hand_result = self.hands.process(RGB_frame)
            
            # Process head pose
            face_result = self.face_mesh.process(RGB_frame)
            
            # Make frame writeable again for drawing
            RGB_frame.flags.writeable = True
            
            self._process_hand_gestures(hand_result, frame)
            self._process_head_pose(face_result, frame)
            
            # Store frame for display (thread-safe)
            if self.show_video:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            
            # No delay - run as fast as possible for better FPS
            # time.sleep(0.01)
    
    def _process_hand_gestures(self, result, frame):
        """Process hand gesture detection."""
        pointer_relative_x = None
        pointer_relative_y = None
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks only if showing video
                if self.show_video:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                
                # Calculate palm center (average of palm landmarks)
                palm_x = sum(hand_landmarks.landmark[i].x for i in self.palm_indices) / len(self.palm_indices)
                palm_y = sum(hand_landmarks.landmark[i].y for i in self.palm_indices) / len(self.palm_indices)
                
                # Calculate average pointer position (index + middle fingertips)
                pointer_x = sum(hand_landmarks.landmark[i].x for i in self.pointer_indices) / len(self.pointer_indices)
                pointer_y = sum(hand_landmarks.landmark[i].y for i in self.pointer_indices) / len(self.pointer_indices)
                
                # Track ONLY the pointer position relative to palm
                pointer_relative_x = pointer_x - palm_x
                pointer_relative_y = pointer_y - palm_y
                
                # Draw visualization (only if show_video is enabled)
                if self.show_video:
                    palm_px = int(palm_x * frame.shape[1])
                    palm_py = int(palm_y * frame.shape[0])
                    pointer_px = int(pointer_x * frame.shape[1])
                    pointer_py = int(pointer_y * frame.shape[0])
                    
                    cv2.circle(frame, (palm_px, palm_py), 8, (0, 255, 0), -1)  # green palm
                    cv2.circle(frame, (pointer_px, pointer_py), 8, (255, 0, 0), -1)  # blue pointers
                    cv2.line(frame, (palm_px, palm_py), (pointer_px, pointer_py), (255, 255, 0), 2)
        
        # Add to positions queue
        current_time = time.time()
        if pointer_relative_x is not None:
            self.positions_queue.append((pointer_relative_x, pointer_relative_y, current_time))
        
        # Calculate velocity using time-weighted smoothing
        if len(self.positions_queue) >= 5:  # need some history
            # Get current and smoothed past position
            current_x, current_y, _ = self.positions_queue[-1]
            
            # Time-weighted average of past positions
            x_sum = y_sum = w_sum = 0
            for px, py, t_stamp in list(self.positions_queue)[:-1]:  # exclude current
                weight = math.exp(-(current_time - t_stamp) / self.tau)
                x_sum += px * weight
                y_sum += py * weight
                w_sum += weight
            
            if w_sum > 0:
                avg_past_x = x_sum / w_sum
                avg_past_y = y_sum / w_sum
                
                # Calculate velocity (change in RELATIVE position)
                oldest_time = self.positions_queue[0][2]
                dt = current_time - oldest_time
                
                if dt > 0:
                    dx = current_x - avg_past_x
                    dy = current_y - avg_past_y
                    
                    vx = dx / dt
                    vy = dy / dt
                    
                    # Gesture detection
                    if current_time - self.last_gesture_time > self.gesture_cooldown:
                        if vy > self.min_gesture_velocity:
                            print("[Hand] DOWN swipe detected")
                            self.gesture_queue.put('DOWN')
                            self.last_gesture_time = current_time
                        elif vy < -self.min_gesture_velocity:
                            print("[Hand] UP swipe detected")
                            self.gesture_queue.put('UP')
                            self.last_gesture_time = current_time
    
    def _process_head_pose(self, result, frame):
        """Process head pose detection."""
        if not result.multi_face_landmarks:
            return
        
        img_height, img_width, _ = frame.shape
        current_time = time.time()
        
        for face_landmarks in result.multi_face_landmarks:
            # Draw face mesh landmarks (simplified for performance)
            if self.show_video:
                # Draw only key facial landmarks instead of full mesh for better performance
                drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_facemesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
            
            # Get key facial landmarks for head pose estimation
            face_3d = []
            face_2d = []
            
            # Key points: nose, left eye, right eye, left mouth, right mouth, chin
            key_points = [1, 33, 263, 61, 291, 199]
            
            for idx in key_points:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # Camera matrix
            focal_length = 1 * img_width
            cam_matrix = np.array([
                [focal_length, 0, img_width / 2],
                [0, focal_length, img_height / 2],
                [0, 0, 1]
            ])
            
            # Distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            if not success:
                return
            
            # Get rotational matrix
            rotmat, _ = cv2.Rodrigues(rot_vec)
            
            # Get angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            
            # Determine head orientation (very strict thresholds - explicit movements only)
            text = "Neutral"
            if x < -12:  # Changed from -8 to -12 (much bigger downward tilt required)
                text = "Looking Down"
            elif x > 15:  # Changed from 12 to 15 (much bigger upward tilt required)
                text = "Looking Up"
            
            # Display orientation
            if self.show_video:
                cv2.putText(frame, f"Head: {text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"X: {int(x)} Y: {int(y)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # NOD detection: trigger when moving from non-neutral back to neutral
            # This matches the original headPoseEstimate.py behavior
            if text != "Neutral":
                # Started a head movement
                if not self.head_pose_active:
                    self.head_pose_active = True
                    self.head_pose_orientation = text
                    self.head_pose_start_time = current_time
                    if self.show_video:
                        cv2.putText(frame, f"Nod detected: {text}", (20, 80), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # Returned to neutral - trigger the gesture if we had a pose
                if self.head_pose_active:
                    # Check if enough time passed (stricter timing window)
                    nod_duration = current_time - self.head_pose_start_time
                    if 0.3 < nod_duration < 1.5:  # Changed from 0.2-2.0s to 0.3-1.5s (more deliberate nod)
                        # Check cooldown
                        if current_time - self.last_gesture_time > self.gesture_cooldown:
                            if self.head_pose_orientation == "Looking Down":
                                print("[Head] DOWN nod detected")
                                self.gesture_queue.put('DOWN')
                                self.last_gesture_time = current_time
                                if self.show_video:
                                    cv2.putText(frame, "NOD DOWN!", (20, 110), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            elif self.head_pose_orientation == "Looking Up":
                                print("[Head] UP nod detected")
                                self.gesture_queue.put('UP')
                                self.last_gesture_time = current_time
                                if self.show_video:
                                    cv2.putText(frame, "NOD UP!", (20, 110), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Reset for next nod
                    self.head_pose_active = False
    
    def display_loop(self):
        """Display video in the main thread. Call this method if show_video=True."""
        if not self.show_video:
            return
        
        while self.running:
            with self.frame_lock:
                if self.latest_frame is not None:
                    cv2.imshow("Hand Tracking", self.latest_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
