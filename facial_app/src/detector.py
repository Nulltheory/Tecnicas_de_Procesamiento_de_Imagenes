# src/detector.py
"""
Detector de landmarks faciales usando MediaPipe.
"""
import cv2
import mediapipe as mp
# Asegúrate de que tu config.py esté en el mismo directorio (src/)
from .config import FACE_MESH_CONFIG, LANDMARK_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar landmarks faciales.
    """
    
    def __init__(self):
        """Inicializa el detector de MediaPipe."""
        config = FACE_MESH_CONFIG.copy()
        # Mantenemos el modo estático para el procesamiento de imágenes
        config['static_image_mode'] = True 
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(**config)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Especificación de dibujo para puntos (usado en "malla")
        self.punto_spec = self.mp_drawing.DrawingSpec(
            color=LANDMARK_COLOR, 
            thickness=LANDMARK_THICKNESS, 
            circle_radius=LANDMARK_RADIUS
        )
    
    def detect(self, image, style: str = "puntos"):
        """
        Detecta landmarks faciales en la imagen con un estilo de dibujo.
        
        Args:
            image (numpy.ndarray): Imagen en formato BGR (OpenCV)
            style (str): "puntos", "malla", o "contornos"
        
        Returns:
            tuple: (imagen_procesada, landmarks, info)
        """
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resultados = self.face_mesh.process(imagen_rgb)
        imagen_con_puntos = image.copy()
        
        info = {
            "rostros_detectados": 0,
            "total_landmarks": 0,
            "deteccion_exitosa": False
        }
        
        if resultados.multi_face_landmarks:
            info["rostros_detectados"] = len(resultados.multi_face_landmarks)
            
            rostro = resultados.multi_face_landmarks[0]
            info["total_landmarks"] = len(rostro.landmark)
            info["deteccion_exitosa"] = True
            
            alto, ancho = image.shape[:2]
            
            # --- LÓGICA DE DIBUJO BASADA EN EL ESTILO ---

            if style == "puntos":
                # Estilo original: dibujar círculos manualmente
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
            
            elif style == "malla":
                # Estilo Malla: Puntos + Teselación
                self.mp_drawing.draw_landmarks(
                    image=imagen_con_puntos,
                    landmark_list=rostro,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.punto_spec, # Dibuja los puntos
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            elif style == "contornos":
                # Estilo Contornos: Solo las líneas principales
                self.mp_drawing.draw_landmarks(
                    image=imagen_con_puntos,
                    landmark_list=rostro,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None, # No dibuja puntos
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            return imagen_con_puntos, rostro, info
        
        # No se detectó rostro
        return imagen_con_puntos, None, info
    
    def close(self):
        """Libera recursos del detector."""
        self.face_mesh.close()