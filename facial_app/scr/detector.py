# src/detector.py
"""
Detector de landmarks faciales usando MediaPipe.
"""
import cv2
import mediapipe as mp
from .config import FACE_MESH_CONFIG, LANDMARK_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar landmarks faciales.
    """
    
    def __init__(self):
        """Inicializa el detector de MediaPipe."""
        # Initialize normally - the issue is in the MediaPipe library itself
        config = FACE_MESH_CONFIG.copy()
        config['static_image_mode'] = True
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(**config)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect(self, image):
        """
        Detecta landmarks faciales en la imagen.
        
        Args:
            image (numpy.ndarray): Imagen en formato BGR (OpenCV)
        
        Returns:
            tuple: (imagen_procesada, landmarks, info)
                - imagen_procesada: imagen con landmarks dibujados
                - landmarks: objeto de landmarks de MediaPipe
                - info: diccionario con información de detección
        """
        # Convertir BGR a RGB para MediaPipe
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen
        resultados = self.face_mesh.process(imagen_rgb)
        
        # Crear copia para dibujar
        imagen_con_puntos = image.copy()
        
        info = {
            "rostros_detectados": 0,
            "total_landmarks": 0,
            "deteccion_exitosa": False
        }
        
        # Si se detectaron rostros
        if resultados.multi_face_landmarks:
            info["rostros_detectados"] = len(resultados.multi_face_landmarks)
            
            # Tomar el primer rostro
            rostro = resultados.multi_face_landmarks[0]
            info["total_landmarks"] = len(rostro.landmark)
            info["deteccion_exitosa"] = True
            
            # Dibujar landmarks
            alto, ancho = image.shape[:2]
            
            for punto in rostro.landmark:
                coord_x_pixel = int(punto.x * ancho)
                coord_y_pixel = int(punto.y * alto)
                
                cv2.circle(
                    imagen_con_puntos,
                    (coord_x_pixel, coord_y_pixel),
                    LANDMARK_RADIUS,
                    LANDMARK_COLOR,
                    LANDMARK_THICKNESS
                )
            
            return imagen_con_puntos, rostro, info
        
        # No se detectó rostro
        return imagen_con_puntos, None, info
    
    def close(self):
        """Libera recursos del detector."""
        self.face_mesh.close()