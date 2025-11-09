import cv2
import mediapipe as mp
import numpy as np
import time


mp_facemesh = mp.solutions.face_mesh = mp.solutions.face_mesh
face_mesh = mp_facemesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
Snap = True
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

capture = cv2.VideoCapture(0) #stores opened webcam

porient = "Neutral"
stop = False
pauseTime = 0
timer = 1

# def snapCheck(cTime, orientation):
#     if orientation != porient:
#         print("swipe")
#         stop = False
#         porient = orientation
#         print(stop)
#     if cTime - pauseTime > timer:
#         stop = False
#         porient = orientation



while capture.isOpened():
    success, image =  capture.read();
    if not success:
        print("Failed to read from webcam")
        continue
    start = time.time()
    #convert image to RGB + flip image for selfie view
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #only read from image
    image.flags.writeable = False
    
    result = face_mesh.process(image)
    
    #reanable wrtiting to image
    image.flags.writeable = True
    #re-convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_height, image_width, image_channel = image.shape
    face_3d = []
    face_2d = []
    
    eyebrows = [70,63,105,66,107,55,65,52,53,46,336,296,334,293,300,283]
    eyes = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,385,386,387,388,466,467,388,466]
    
    def eye_distance(landmarks, eyebrow_points, eye_points, image_height, image_width):

        eyebrow_mean = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] for p in eyebrow_points], axis=0)
        eye_mean     = np.mean([[landmarks[p].x, landmarks[p].y, landmarks[p].z] for p in eye_points], axis=0)

        raw_distance = np.linalg.norm(eye_mean - eyebrow_mean)

        left_eye_center  = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye_center = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
        inter_eye_dist   = np.linalg.norm(left_eye_center - right_eye_center)

        if inter_eye_dist == 0:
            return 0.0
        
        normalized_distance = raw_distance / inter_eye_dist
        return normalized_distance
        
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark): #lm contains the index of nose ears eyes etc
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * image_width, lm.y * img_height)
                        nose_3d = (lm.x * image_width, lm.y * img_height, lm.z * 3000)
                    
                    x,y =  int(lm.x * image_width), int(lm.y * img_height)
                    
                    face_2d.append([x,y])
                    face_3d.append([x,y, lm.z]) #z is depth
                    
            face_2d = np.array(face_2d, dtype=np.float64) #convert to numpy array
            face_3d = np.array(face_3d, dtype=np.float64)
            
            #camera matrix/camera calibration
            focal_length = 1 * image_width
            
            cam_matrix = np.array([[focal_length, 0, image_width / 2],
                                   [0, focal_length, img_height / 2],
                                   [0, 0, 1]])
            #distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            #estimate image projections
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            #get rotational matrix, jacobian matrx (not used)
            rotmat, jacmat = cv2.Rodrigues(rot_vec)
            
            #get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            current_time = time.time()
            
            if y < -10:
                text = "Looking Left"
                if stop == False:
                    pauseTime = time.time()
                    stop = True
                    porient = text
            elif y > 10: 
                text = "Looking Right"
                if stop == False:
                    pauseTime = time.time()
                    stop = True
                    porient = text
            elif x < -3:
                text = "Looking Down"
                if stop == False:
                    pauseTime = time.time()
                    stop = True
                    porient = text
            elif x > 7:
                text = "Looking Up"
                if stop == False:
                    pauseTime = time.time()
                    stop = True
                    porient = text
            else:
               text = "Neutral"
            
            # if text != "Neutral":
            #     porient = text
                
            if stop == True:
                # snapCheck(current_time, text)
                if text != porient:
                    match porient:
                        case "Looking Left":
                            res = "Swipe Left"
                        case "Looking Right":
                            res = "Swipe Right"
                        case "Looking Up":
                            res = "Swipe Up"
                        case "Looking Down":
                            res = "Swipe Down"
                    cv2.putText(image, res, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
                    stop = False
                if current_time - pauseTime > timer:
                    stop = False
                
                
                

            
            
            nose_3d_proj, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255,0,0), 3)
            
            #add text
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  
            if(Snap):
                avgdst = eye_distance(face_landmarks.landmark, eyebrows, eyes, img_height, image_width)
                Snap = False;
            
            dist = eye_distance(face_landmarks.landmark, eyebrows, eyes, img_height, image_width)
            cv2.putText(image, "Eye Dist: " + str(np.round(dist,2)), (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, "Avg: " + str(np.round(avgdst,2)), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
            if text == "Neutral":
                if (avgdst - dist) < -0.05:
                    text2 = "Like"
                else:
                    text2 = "None"
            else:
                text2 = "None"
            cv2.putText(image, text2, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            
            
        end = time.time()
        totalTime = 1
        if( end != start):
            totalTime = end - start
        
        fps = 1 / totalTime
        ##print("fps:", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_facemesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    
    cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
capture.release()