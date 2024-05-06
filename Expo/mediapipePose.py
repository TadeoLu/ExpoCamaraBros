import cv2
import mediapipe as mp
import time

# Inicializar el estimador de postura
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Crear objeto de captura
cap = cv2.VideoCapture(0)

# Función para capturar el promedio de las coordenadas
def capturar_promedio(pose, cap):
    landmark_0_x = 0
    landmark_0_y = 0
    landmark_31_x = 0
    landmark_31_y = 0
    landmark_32_x = 0
    landmark_32_y = 0
    num_frames = 0

    start_time = time.time()

    while time.time() - start_time < 5:
        _, frame = cap.read()

        try:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                landmark_0_x += landmarks[0].x
                landmark_0_y += landmarks[0].y
                landmark_31_x += landmarks[31].x
                landmark_31_y += landmarks[31].y
                landmark_32_x += landmarks[32].x
                landmark_32_y += landmarks[32].y
                num_frames += 1

        except Exception as e:
            break

    if num_frames > 0:
        landmark_0_x /= num_frames
        landmark_0_y /= num_frames
        landmark_31_x /= num_frames
        landmark_31_y /= num_frames
        landmark_32_x /= num_frames
        landmark_32_y /= num_frames

    return landmark_0_x, landmark_0_y, landmark_31_x, landmark_31_y, landmark_32_x, landmark_32_y

# Capturar el promedio de coordenadas
promedio_0_x, promedio_0_y, promedio_31_x, promedio_31_y, promedio_32_x, promedio_32_y = capturar_promedio(pose, cap)

while cap.isOpened():
    _, frame = cap.read()

    try:
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(RGB)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            new_landmark_31_x = landmarks[31].x
            new_landmark_32_x = landmarks[32].x
            new_landmark_0_y = landmarks[0].y
            promedio_31_x_rest = promedio_31_x * 0.05
            promedio_32_x_rest = promedio_32_x * 0.05
            promedio_0_y_rest = promedio_0_y * 0.4
            promedio_0_y_rest2 = promedio_0_y * 0.1
            if (new_landmark_31_x - promedio_31_x) >= promedio_31_x_rest and (new_landmark_32_x - promedio_32_x) >= promedio_32_x_rest :
                
                if (new_landmark_0_y - promedio_0_y) >= promedio_0_y_rest:
                    print("Movimiento izquierda agacharse")
                elif (new_landmark_0_y - promedio_0_y) <= -promedio_0_y_rest2:
                    print("Movimiento izquierda salto ")
                else:
                    print ("movimiento a la izquierda")
            elif (new_landmark_31_x - promedio_31_x) <= -promedio_31_x_rest and (new_landmark_32_x - promedio_32_x) <= -promedio_32_x_rest :
                if (new_landmark_0_y - promedio_0_y) >= promedio_0_y_rest:
                    print("Movimiento derecha agacharse")
                elif (new_landmark_0_y - promedio_0_y) <= -promedio_0_y_rest2:
                    print("Movimiento derecha salto ")
                else: 
                    print ("movimiento a la derecha")

            elif (new_landmark_0_y - promedio_0_y) >= promedio_0_y_rest:
                print("Movimiento agacharse")
            elif (new_landmark_0_y - promedio_0_y) <= -promedio_0_y_rest2:
                print("Movimiento salto ")
            else:
                print("estas en el medio")
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Output', frame)

    except Exception as e:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
