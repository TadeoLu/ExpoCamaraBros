import cv2  # Importamos la biblioteca OpenCV para procesamiento de imágenes y vídeo
import mediapipe as mp  # Importamos Mediapipe para el seguimiento de postura
import time  # Importamos la biblioteca para medir el tiempo
import pyautogui  # Importamos PyAutoGUI para simular entradas de teclado
import numpy as np

# Inicializar el estimador de postura
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)

# Creamos una instancia del estimador de postura con ciertas configuraciones de confianza
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializamos la captura de vídeo desde la cámara (cámara número 0)
cap = cv2.VideoCapture(0)

# Función para capturar el promedio de las coordenadas de ciertos puntos clave
def capturar_promedio(pose, cap):
    # Inicializamos variables para el promedio y el conteo de frames
    nariz_y = 0
    pie_derecha_x = 0
    pie_izquierda_x = 0
    num_frames = 0
    mano_derecha_y = 0.2
    mano_derecha_x = 0
    mano_izquierda_y = 0.2
    mano_izquierda_x = 0

    # Obtenemos el tiempo actual
    start_time = time.time()

    # Mientras no hayan pasado 5 segundos desde el inicio
    while time.time() - start_time < 5:
        _, frame = cap.read()  # Capturamos un frame desde la cámara

        try:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertimos el frame a formato RGB
            results = pose.process(RGB)  # Procesamos el frame para detectar la postura

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Acumulamos las coordenadas de ciertos puntos clave
                nariz_y += landmarks[0].y
                pie_derecha_x += landmarks[31].x
                pie_izquierda_x += landmarks[32].x
                mano_derecha_y += landmarks[20].y
                mano_derecha_x += landmarks[20].x
                mano_izquierda_y += landmarks[19].y
                mano_izquierda_x += landmarks[19].x
                num_frames += 1

        except Exception as e:
            break  # Salimos del bucle si ocurre una excepción

    # Calculamos el promedio de las coordenadas si se capturaron frames
    if num_frames > 0:
        nariz_y /= num_frames
        pie_derecha_x /= num_frames
        pie_izquierda_x /= num_frames
        mano_derecha_y /= num_frames
        mano_derecha_x /= num_frames
        mano_izquierda_y /= num_frames
        mano_izquierda_x /= num_frames

    return nariz_y, pie_derecha_x, pie_izquierda_x, mano_derecha_x, mano_derecha_y, mano_izquierda_y, mano_izquierda_x

# Obtenemos los promedios de las coordenadas llamando a la función
promedio_nariz_y, promedio_pie_derecha_x, promedio_pie_izquierda_x, promedio_mano_derecha_x, promedio_mano_derecha_y, promedio_mano_izquierda_y, promedio_mano_izquierda_x= capturar_promedio(pose, cap)

# Inicializamos variables para rastrear el estado anterior y si ya se realizó una acción
estado_anterior = "centro"
ha_dicho_una_vez = False
estado_actual = "centro"
# Bucle principal para procesar el video de la cámara
while cap.isOpened():
    _, frame = cap.read()  # Capturamos un frame desde la cámara

    try:
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertimos el frame a formato RGB
        results = pose.process(RGB)  # Procesamos el frame para detectar la postura
        numpy_frame_from_opencv = np.array(frame)
        current_time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        recognizer.recognize_async(mp_image, current_time_ms)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            new_pie_derecha_x = landmarks[31].x
            new_pie_izquierda_x = landmarks[32].x
            new_nariz_y = landmarks[0].y
            new_mano_derecha_y = landmarks[20].y
            new_mano_derecha_x = landmarks[20].x
            new_mano_izquierda_y = landmarks[19].y
            new_mano_izquierda_x = landmarks[19].x
            # Definimos umbrales para determinar acciones
            promedio_pie_derecha_x_rest = promedio_pie_derecha_x * 0.08
            promedio_pie_izquierda_x_rest = promedio_pie_izquierda_x * 0.08
            promedio_nariz_y_rest_agarcharse = promedio_nariz_y * 0.4
            promedio_nariz_y_rest_saltar = promedio_nariz_y * 0.2
            promedio_mano_derecha_y_rest = promedio_mano_derecha_y * 0.2
            promedio_mano_izquierda_y_rest = promedio_mano_izquierda_y * 0.2
            #Boton
            # promedio_mano_derecha_x_rest = promedio_mano_derecha_x * 0.1
            # promedio_mano_derecha_x_rest2 = promedio_mano_derecha_x * 0.15
            # promedio_mano_izquierda_x_rest = promedio_mano_izquierda_x * 0.1
            # promedio_mano_izquierda_x_rest2 = promedio_mano_izquierda_x * 0.15

            # Determinamos el estado actual
            if (new_mano_derecha_y - promedio_mano_derecha_y) <= -promedio_mano_derecha_y_rest:
                if not ha_dicho_una_vez:
                    ha_dicho_una_vez = True
                    print("Movimiento de mano derecha")
                    pyautogui.press('o')  
                # if(new_mano_derecha_x - promedio_mano_derecha_x) >= promedio_mano_derecha_x_rest:
                #     print("boton izquierda")
                #     pyautogui.press('i')  
                # elif(new_mano_derecha_x - promedio_mano_derecha_x) <= -promedio_mano_derecha_x_rest:
                #     print("Boton derecha")
                #     pyautogui.press('p')  
                # elif (new_mano_derecha_x - promedio_mano_derecha_x) <= promedio_mano_derecha_x_rest2 and (new_mano_derecha_x - promedio_mano_derecha_x) >= -promedio_mano_derecha_x_rest2:
                #     print("Boton centro")
                #     pyautogui.press('o') 
            elif (new_mano_izquierda_y - promedio_mano_izquierda_y) <= -promedio_mano_izquierda_y_rest:
                if not ha_dicho_una_vez:
                    ha_dicho_una_vez = True
                    print("Movimiento de mano izquierda")
                    pyautogui.press('i') 
            elif (new_nariz_y - promedio_nariz_y) >= promedio_nariz_y_rest_agarcharse:
                estado_actual = "agacharse"
                if not ha_dicho_una_vez:
                    pyautogui.press('s')  # Simulamos la tecla 's' si no se ha realizado una acción previamente
                    ha_dicho_una_vez = True

            elif (new_nariz_y - promedio_nariz_y) <= -promedio_nariz_y_rest_saltar:
                estado_actual = "salto"
                if not ha_dicho_una_vez:
                    pyautogui.press('w')  # Simulamos la tecla 'w' si no se ha realizado una acción previamente
                    ha_dicho_una_vez = True

            # elif (new_pie_derecha_x - promedio_pie_derecha_x) >= promedio_pie_derecha_x_rest and (new_pie_izquierda_x - promedio_pie_izquierda_x) >= promedio_pie_izquierda_x_rest:
            #     estado_actual = "izquierda"
            #     if not ha_dicho_una_vez and (estado_anterior != "agacharse" or estado_anterior != "salto"):
            #         pyautogui.press('a')  # Simulamos la tecla 'a' si no se ha realizado una acción previamente
            #         ha_dicho_una_vez = True

            # elif (new_pie_derecha_x - promedio_pie_derecha_x) <= -promedio_pie_derecha_x_rest and (new_pie_izquierda_x - promedio_pie_izquierda_x) <= -promedio_pie_izquierda_x_rest:
            #     estado_actual = "derecha"
            #     if not ha_dicho_una_vez and (estado_anterior != "agacharse" or estado_anterior != "salto"):
            #         pyautogui.press('d')  # Simulamos la tecla 'd' si no se ha realizado una acción previamente
            #         ha_dicho_una_vez = True

            else:
                estado_actual = "centro"
                # Simulamos una acción si se ha movido de izquierda a centro o de derecha a centro
                # if estado_anterior == "izquierda" and estado_actual == "centro":
                #     print("me movi de izquierda a centro")
                #     estado_actual = "me movi de izquierda a centro"
                #     pyautogui.press('d')  # Simulamos la tecla 'd'
                # elif estado_anterior == "derecha" and estado_actual == "centro":
                #     print("me movi de derecha a centro")
                #     estado_actual = "me movi de derecha a centro"
                #     pyautogui.press('a')  # Simulamos la tecla 'a'
                    
            # Actualizamos el estado anterior y el indicador de acción realizada
            if estado_actual != estado_anterior:
                ha_dicho_una_vez = False
                print(f"Nuevo estado: {estado_actual}")
                estado_anterior = estado_actual

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Output', frame)  # Mostramos el frame con dibujos de puntos clave y conexiones

    except Exception as e:
        print("Excepcion:", e)
        break  # Salimos del bucle si ocurre una excepción

    if cv2.waitKey(1) == ord('q'):
        break  # Salimos del bucle si se presiona la tecla 'q'

# Liberamos la cámara y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()