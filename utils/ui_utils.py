import streamlit as st
from PIL import Image
import io
import time

def mostrar_progreso(mensaje: str, segundos: int = 2):
    with st.spinner(mensaje):
        time.sleep(segundos)

def mostrar_resultado_con_descarga(imagen: Image.Image):
    st.subheader("✅ Resultado final")
    st.image(imagen, caption="Resultado restaurado", width=512)
    buf = io.BytesIO()
    imagen.save(buf, format="PNG")
    st.download_button(
        label="📥 Descargar imagen restaurada",
        data=buf.getvalue(),
        file_name="restaurada.png",
        mime="image/png"
    )
