# src/config.py
"""
Configuración del detector de landmarks faciales.
"""

# Parámetros del modelo MediaPipe
FACE_MESH_CONFIG = {
    "static_image_mode": True,
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5
}

# Configuración de visualización
LANDMARK_COLOR = (0, 255, 0)  # Verde en BGR
LANDMARK_RADIUS = 2
LANDMARK_THICKNESS = -1  # Relleno

# Cantidad de landmarks esperados
TOTAL_LANDMARKS = 478
