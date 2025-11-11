"""
🛠️ Utilidades de UI para la aplicación de restauración fotográfica

Este módulo contiene funciones auxiliares para mejorar la experiencia de usuario
en la interfaz de Streamlit.
"""

import streamlit as st
import io
from PIL import Image


def mostrar_progreso(operacion: str, progreso: int):
    """Muestra un progreso visual en Streamlit"""
    if progreso < 100:
        st.progress(progreso / 100)
        st.info(f"🔄 {operacion}: {progreso}%")

def mostrar_resultado_con_descarga(imagen: Image.Image, nombre_archivo: str, titulo: str = "Imagen Restaurada"):
    """Muestra resultado con opciones de descarga"""
    
    # Preparar opciones de descarga
    col1, col2 = st.columns(2)
    
    with col1:
        # Descarga JPEG
        buffer = io.BytesIO()
        imagen.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        st.download_button(
            label="📥 Descargar JPEG (Alta calidad)",
            data=buffer.getvalue(),
            file_name=f"restaurada_{nombre_archivo}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    with col2:
        # Descarga PNG
        buffer_png = io.BytesIO()
        imagen.save(buffer_png, format='PNG')
        buffer_png.seek(0)

        st.download_button(
            label="📥 Descargar PNG (Sin pérdida)",
            data=buffer_png.getvalue(),
            file_name=f"restaurada_{nombre_archivo}.png",
            mime="image/png",
            use_container_width=True
        )

def mostrar_info_sistema(torch_available: bool, modules_available: bool):
    """Muestra información del estado del sistema"""
    with st.expander("🔧 Información del Sistema", expanded=False):
        if torch_available and modules_available:
            st.success("✅ Sistema completo - Todas las funciones disponibles")
            st.info("• Modelos: Real-ESRGAN, CodeFormer, GFPGAN, Stable Diffusion")
            st.info("• Análisis: CLIP + Gemini AI")
            st.info("• Procesamiento: GPU/CPU optimizado")
        elif torch_available:
            st.warning("⚠️ PyTorch disponible - Funciones limitadas")
            st.info("• Modelos básicos disponibles")
            st.info("• Análisis limitado")
        elif modules_available:
            st.info("ℹ️ Funciones básicas disponibles - Sin PyTorch")
            st.info("• Procesamiento con OpenCV")
            st.info("• Sin análisis de IA")
        else:
            st.error("❌ Sistema básico - Funciones mínimas")
            st.info("• Solo procesamiento de imagen básico")
            st.info("• Sin análisis automático")


def validar_imagen_subida(file_uploader_result) -> bool:
    """Valida que la imagen subida sea válida"""
    if not file_uploader_result:
        return False
    
    try:
        # Verificar que se puede abrir como imagen
        image = Image.open(file_uploader_result)
        # Verificar dimensiones válidas
        if image.size[0] < 10 or image.size[1] < 10:
            st.error("⚠️ Imagen demasiado pequeña (mínimo 10x10 píxeles)")
            return False
        if image.size[0] > 4000 or image.size[1] > 4000:
            st.warning("⚠️ Imagen muy grande - se redimensionará automáticamente")
        
        return True
    except Exception:
        st.error("❌ No se pudo procesar la imagen. Verifica que sea un archivo JPG/PNG válido.")
        return False


def mostrar_estadisticas_procesamiento(algoritmos_aplicados: int, tiempo_procesamiento: float = None):
    """Muestra estadísticas del procesamiento realizado"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🖼️ Algoritmos", algoritmos_aplicados, "aplicados")
    
    with col2:
        if tiempo_procesamiento:
            st.metric("⏱️ Tiempo", f"{tiempo_procesamiento:.1f}s", "procesamiento")
        else:
            st.metric("⏱️ Tiempo", "Rápido", "optimizado")
    
    with col3:
        st.metric("📊 Calidad", "Alta", "esperada")


def crear_seccion_analisis(usar_analisis: bool, resultados: dict = None):
    """Crea una sección de análisis de resultados"""
    if not usar_analisis:
        return
    
    with st.expander("📊 Análisis de Resultados", expanded=True):
        if resultados:
            for key, value in resultados.items():
                st.write(f"**{key}:** {value}")
        else:
            st.info("No hay resultados de análisis disponibles")


def mostrar_errores_graciosos(operacion: str, error: Exception):
    """Muestra errores de manera user-friendly"""
    st.error(f"❌ Error en {operacion}: {str(error)}")
    st.info("💡 La aplicación continuará con las funciones disponibles")


