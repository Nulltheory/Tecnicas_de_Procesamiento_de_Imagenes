"""
Aplicación Streamlit para detección de landmarks faciales.
(Versión con Estilos, JSON y Análisis de Expresiones)
"""
import streamlit as st
from PIL import Image
import io
import json

from src.detector import FaceLandmarkDetector
from src.utils import pil_to_cv2, cv2_to_pil, resize_image
from src.config import TOTAL_LANDMARKS


# --- 1. CACHE DEL RECURSO (El Modelo) ---
@st.cache_resource
def cargar_detector():
    """Carga el detector de landmarks una sola vez."""
    print("Cargando modelo de MediaPipe...")
    detector = FaceLandmarkDetector()
    return detector

# --- 2. FUNCIONES DE ANÁLISIS DE EXPRESIÓN (OPCIÓN B) ---

def calcular_apertura_boca(landmarks, alto):
    """Calcula la distancia vertical entre los labios."""
    if landmarks is None:
        return 0
    # Landmark 13: Centro del labio superior
    # Landmark 14: Centro del labio inferior
    punto_superior = landmarks.landmark[13]
    punto_inferior = landmarks.landmark[14]
    
    y1_px = punto_superior.y * alto
    y2_px = punto_inferior.y * alto
    
    distancia = abs(y2_px - y1_px)
    return distancia

def calcular_apertura_ojos(landmarks, alto):
    """Calcula la apertura promedio (vertical) de ambos ojos."""
    if landmarks is None:
        return 0
    
    # Ojo Izquierdo (Landmarks 159, 145)
    punto_sup_izq = landmarks.landmark[159]
    punto_inf_izq = landmarks.landmark[145]
    y_sup_izq = punto_sup_izq.y * alto
    y_inf_izq = punto_inf_izq.y * alto
    dist_izq = abs(y_sup_izq - y_inf_izq)
    
    # Ojo Derecho (Landmarks 386, 374)
    punto_sup_der = landmarks.landmark[386]
    punto_inf_der = landmarks.landmark[374]
    y_sup_der = punto_sup_der.y * alto
    y_inf_der = punto_inf_der.y * alto
    dist_der = abs(y_sup_der - y_inf_der)
    
    # Promedio
    distancia_promedio = (dist_izq + dist_der) / 2
    return distancia_promedio

# --- 3. FUNCIÓN PARA CONVERTIR LANDMARKS A DICCIONARIO (OPCIÓN D) ---
def landmarks_to_dict(landmarks, alto, ancho):
    """Convierte landmarks a diccionario exportable."""
    if landmarks is None:
        return {}
        
    data = []
    for idx, punto in enumerate(landmarks.landmark):
        data.append({
            "id": idx,
            "x_px": punto.x * ancho,
            "y_px": punto.y * alto,
            "z_rel": punto.z,
            "x_norm": punto.x,
            "y_norm": punto.y
        })
    return {"total_landmarks": len(data), "points": data}

# --- 4. CACHE DE DATOS (El Resultado) ---
@st.cache_data
def procesar_imagen_cacheada(file_bytes: bytes, style: str):
    """
    Procesa la imagen, calcula métricas y cachea el resultado.
    """
    print(f"Procesando imagen con estilo: {style}") 
    
    detector = cargar_detector()
    
    imagen_original = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    imagen_cv2 = pil_to_cv2(imagen_original)
    
    imagen_cv2_redim = resize_image(imagen_cv2, max_width=800)
    alto, ancho = imagen_cv2_redim.shape[:2]
    
    style_map = {
        "Puntos": "puntos",
        "Puntos + Malla": "malla",
        "Solo Contornos": "contornos"
    }
    style_arg = style_map.get(style, "puntos")
    
    imagen_procesada, landmarks_raw, info = detector.detect(imagen_cv2_redim, style=style_arg)
    
    # --- ¡NUEVO! CÁLCULO DE MÉTRICAS (OPCIÓN B) ---
    if info["deteccion_exitosa"]:
        info["apertura_boca_px"] = calcular_apertura_boca(landmarks_raw, alto)
        info["apertura_ojos_px"] = calcular_apertura_ojos(landmarks_raw, alto)
    else:
        info["apertura_boca_px"] = 0
        info["apertura_ojos_px"] = 0
    
    # Convertir a PIL
    pil_original = cv2_to_pil(imagen_cv2_redim)
    pil_procesada = cv2_to_pil(imagen_procesada)
    
    # Convertir PIL A BYTES PNG
    bytes_original = io.BytesIO()
    pil_original.save(bytes_original, format="PNG")
    
    bytes_procesada = io.BytesIO()
    pil_procesada.save(bytes_procesada, format="PNG")

    # Crear el diccionario para JSON (OPCIÓN D)
    json_data = landmarks_to_dict(landmarks_raw, alto, ancho)
    json_string = json.dumps(json_data, indent=2)

    return bytes_procesada.getvalue(), info, bytes_original.getvalue(), json_string


# --- 5. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Detector de Landmarks Faciales",
    layout="wide"
)

# Título y descripción
st.title("Detector de Landmarks Faciales")
st.markdown("""
Esta aplicación detecta **478 puntos clave** en rostros humanos usando MediaPipe.
Subí una imagen con un rostro y mirá la magia de la visión por computadora.
""")

# --- 6. SIDEBAR CON ESTILOS (OPCIÓN A) ---
with st.sidebar:
    st.header("Opciones de Visualización")
    estilo_visual = st.radio(
        "Elige un estilo:",
        ["Puntos", "Puntos + Malla", "Solo Contornos"],
        key="estilo_radio"
    )

    st.divider()
    st.header("Información")
    st.markdown("""
    ### ¿Qué son los Landmarks?
    Son puntos de referencia que mapean:
    - Ojos (iris, párpados)
    - Nariz (puente, fosas)
    - Boca (labios, comisuras)
    - Contorno facial
    
    ### Aplicaciones
    - Filtros AR (Instagram)
    - Análisis de expresiones
    - Animación facial
    - Autenticación biométrica
    """)
    
    st.divider()
    st.caption("Desarrollado en el Laboratorio 2 - IFTS24")

# Uploader de imagen
uploaded_file = st.file_uploader(
    "Subí una imagen con un rostro",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG",
    key="uploader"
)

# --- 7. LÓGICA DE PROCESAMIENTO AUTOMÁTICA ---
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    with st.spinner("Detectando landmarks faciales..."):
        bytes_procesada, info, bytes_original, landmarks_json = procesar_imagen_cacheada(bytes_data, estilo_visual)

    # --- 8. MOSTRAR RESULTADOS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(bytes_original)
    
    with col2:
        st.subheader("Landmarks Detectados")
        st.image(bytes_procesada)
    
    st.divider()
    
    if info["deteccion_exitosa"]:
        st.success("Detección exitosa")
        
        # --- ¡NUEVO! MÉTRICAS DE ANÁLISIS (OPCIÓN B) ---
        st.subheader("Análisis de Expresión (en píxeles)")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Apertura de Boca (vertical)", f"{info['apertura_boca_px']:.2f} px")
        with metric_col2:
            st.metric("Apertura de Ojos (promedio)", f"{info['apertura_ojos_px']:.2f} px")
        
        st.divider()

        # Métricas de Detección
        st.subheader("Estadísticas de Detección")
        metric_col3, metric_col4, metric_col5 = st.columns(3)
        
        with metric_col3:
            st.metric("Rostros detectados", info["rostros_detectados"])
        
        with metric_col4:
            st.metric("Landmarks detectados", f"{info['total_landmarks']}/{TOTAL_LANDMARKS}")
        
        with metric_col5:
            porcentaje = (info['total_landmarks'] / TOTAL_LANDMARKS) * 100
            st.metric("Precisión", f"{porcentaje:.1f}%")
        
        st.divider()
        
        # --- BOTÓN DE DESCARGA JSON (OPCIÓN D) ---
        st.download_button(
            label="Descargar Landmarks (JSON)",
            data=landmarks_json,
            file_name=f"{uploaded_file.name}_landmarks.json",
            mime="application/json"
        )
        
    else:
        st.error("No se detectó ningún rostro en la imagen")
        st.info("""
        **Consejos**:
        - Asegurate de que el rostro esté bien iluminado
        - El rostro debe estar mirando hacia la cámara
        - Probá con una imagen de mayor calidad
        """)

else:
    # Mensaje de bienvenida (solo si no hay archivo subido)
    st.info("Subí una imagen para comenzar la detección")
    
    # Ejemplo visual
    st.markdown("### Ejemplo de Resultado")
    st.image(
        "https://ai.google.dev/static/mediapipe/images/solutions/face_landmarker_keypoints.png?hl=es-419", 
        caption="MediaPipe detecta 478 landmarks faciales",
        width=400
    )