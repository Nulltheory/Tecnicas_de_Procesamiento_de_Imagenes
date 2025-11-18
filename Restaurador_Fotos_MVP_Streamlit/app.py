"""
🖼️ app.py
Restauración de imágenes con IA + análisis comparativo
Muestra progreso paso a paso y permite descargar resultados
"""

import io
import streamlit as st
from PIL import Image
from models.diffusion import restaurar_imagen_gfpgan, upscale_imagen_realesrgan, restaurar_imagen_sd, get_model_status
from models.analysis import analizar_imagen_completo

st.set_page_config(page_title="Restauración Fotográfica IA", layout="wide")
st.title("🖼️ Restauración Fotográfica Avanzada (Paso a Paso)")
st.write("GFPGAN → Real-ESRGAN → Stable Diffusion + ControlNet")

hf_token = st.secrets.get("HF_TOKEN", "")
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen original", use_column_width=True)

    if st.button("🚀 Restaurar Imagen"):
        try:
            original_bytes = uploaded_file.getvalue()

            # 1️⃣ GFPGAN - Restauración facial
            with st.spinner("1/3 GFPGAN: Restaurando rostros y detalles..."):
                img_gfpgan = restaurar_imagen_gfpgan(img)
            st.image(img_gfpgan, caption="Paso 1: GFPGAN", use_column_width=True)

            # 2️⃣ Real-ESRGAN - Upscale y mejora de detalles
            with st.spinner("2/3 Real-ESRGAN: Mejorando resolución y detalles..."):
                img_esrgan = upscale_imagen_realesrgan(img_gfpgan)
            st.image(img_esrgan, caption="Paso 2: Real-ESRGAN", use_column_width=True)

            # 3️⃣ Stable Diffusion + ControlNet - Restauración creativa
            if hf_token:
                with st.spinner("3/3 Stable Diffusion + ControlNet: Restauración avanzada..."):
                    img_sd = restaurar_imagen_sd(img_esrgan, hf_token)
                st.image(img_sd, caption="Paso 3: Stable Diffusion", use_column_width=True)
            else:
                st.info("HF_TOKEN no configurado, omitiendo SD+ControlNet")
                img_sd = img_esrgan

            # 📦 Imagen final
            st.subheader("✅ Imagen final restaurada")
            st.image(img_sd, use_column_width=True)

            # Botón para descargar
            buf = io.BytesIO()
            img_sd.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button("💾 Descargar Imagen Restaurada", data=byte_im, file_name="restaurada.jpg", mime="image/jpeg")

            # 🔍 Análisis de calidad (opcional)
            if gemini_api_key:
                with st.spinner("Analizando calidad de restauración con Gemini + CLIP..."):
                    restored_bytes = buf.getvalue()
                    resultados = analizar_imagen_completo(
                        original_bytes, restored_bytes, gemini_api_key
                    )

                st.subheader("📊 Resultados de Análisis")
                # CLIP
                clip_result = resultados.get("clip_clasificacion")
                if clip_result:
                    st.write("**Clasificación CLIP:**")
                    st.json(clip_result)

                # Gemini comparativo
                gemini_result = resultados.get("gemini_analisis")
                if gemini_result:
                    st.write("**Informe Gemini comparativo:**")
                    st.write(gemini_result)
            else:
                st.info("Provee GEMINI_API_KEY para análisis de IA comparativo.")

        except Exception as e:
            st.error(f"Error durante restauración/análisis: {e}")

# Sidebar: estado de modelos
st.sidebar.header("Estado de los modelos")
status = get_model_status()
for model, state in status.items():
    st.sidebar.write(f"{model.upper()}: {state}")
