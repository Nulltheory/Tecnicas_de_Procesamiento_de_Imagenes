# src/utils.py
"""
Funciones auxiliares para procesamiento de imágenes.
"""
import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image):
    """
    Convierte una imagen PIL a formato OpenCV (numpy array BGR).
    
    Args:
        pil_image (PIL.Image): Imagen en formato PIL
    
    Returns:
        numpy.ndarray: Imagen en formato OpenCV (BGR)
    """
    # Convertir PIL a RGB numpy array
    rgb_array = np.array(pil_image.convert('RGB'))
    # Convertir RGB a BGR (formato OpenCV)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(cv2_image):
    """
    Convierte una imagen OpenCV a formato PIL.
    
    Args:
        cv2_image (numpy.ndarray): Imagen en formato OpenCV (BGR)
    
    Returns:
        PIL.Image: Imagen en formato PIL (RGB)
    """
    # Convertir BGR a RGB
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convertir a PIL
    pil_image = Image.fromarray(rgb_array)
    return pil_image


def resize_image(image, max_width=800):
    """
    Redimensiona la imagen manteniendo el aspect ratio.
    
    Args:
        image (numpy.ndarray): Imagen OpenCV
        max_width (int): Ancho máximo deseado
    
    Returns:
        numpy.ndarray: Imagen redimensionada
    """
    alto, ancho = image.shape[:2]
    
    if ancho > max_width:
        ratio = max_width / ancho
        nuevo_ancho = max_width
        nuevo_alto = int(alto * ratio)
        image = cv2.resize(image, (nuevo_ancho, nuevo_alto))
    
    return image