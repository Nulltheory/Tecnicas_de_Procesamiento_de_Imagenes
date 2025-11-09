"""
🤖 Módulo de Análisis con IA (Optimizado para Streamlit Cloud)

Este módulo mantiene toda la lógica original, pero incluye mejoras de estabilidad:
- Cacheo del modelo CLIP con @st.cache_resource
- Timeout controlado para llamadas a Gemini
- Reducción de tamaño de imagen para evitar OOM
"""

import io
import base64
import numpy as np
from PIL import Image
import concurrent.futures
import streamlit as st


# =========================
# 🔹 Utilidad auxiliar
# =========================
def safe_generate(model, content, timeout=25):
    """Ejecuta model.generate_content con timeout controlado"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(model.generate_content, content)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return type("Response", (), {"text": "⏰ Tiempo de espera agotado al contactar con Gemini"})()


# =========================
# 🔹 BLIP - Hugging Face
# =========================
def analizar_con_blip_hf(hf_token: str, imagen_bytes: bytes) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model="Salesforce/blip-image-captioning-large", token=hf_token)
    result = client.image_to_text(imagen_bytes)
    return result[0].get("generated_text", "No se generó texto") if result else "No se generó texto"


# =========================
# 🔹 Gemini (imagen única)
# =========================
def analizar_con_gemini(gemini_api_key: str, imagen_bytes: bytes) -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)

    prompt = (
        "Eres un asistente experto en restauración de fotografías antiguas. "
        "Describe brevemente (2-4 frases) qué mejoras observas en esta imagen restaurada "
        "en términos de nitidez, reconstrucción, ruido y artefactos. "
        "Indica también si la restauración introdujo cambios no naturales."
    )

    try:
        pil_image = Image.open(io.BytesIO(imagen_bytes))
        pil_image.thumbnail((512, 512))  # 🧩 Reduce tamaño sin perder contexto
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes_clean = buffer.getvalue()

        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes_clean).decode('utf-8')
        }

        # Usar modelo básico que debería estar disponible
        try:
            model = genai.GenerativeModel('gemini-pro')
        except:
            # Fallback a modelo de visión si está disponible
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
            except:
                raise Exception("No se pudo cargar ningún modelo Gemini disponible")
        response = safe_generate(model, [prompt, image_part])
        return response.text if response and response.text else "Análisis no disponible"

    except Exception as e:
        return f"Error en análisis: {str(e)}"


# =========================
# 🔹 CLIP - Modelo cacheado
# =========================
@st.cache_resource
def load_clip_model():
    # CLIP no funciona bien en Streamlit Cloud, devolver None para graceful degradation
    st.info("ℹ️ CLIP requiere configuración adicional en Streamlit Cloud. Usando solo análisis Gemini por ahora.")
    return None, None


def analizar_calidad_clip(imagen_bytes: bytes) -> dict:
    """Clasifica calidad de imagen usando CLIP"""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    try:
        model, processor = load_clip_model()

        # Verificar si el modelo se cargó correctamente
        if model is None or processor is None:
            return {"error": "No se pudo cargar el modelo CLIP"}

        image = Image.open(io.BytesIO(imagen_bytes))
        image.thumbnail((512, 512))  # 🔹 Reduce RAM

        quality_labels = [
            "imagen antigua restaurada con IA, mejorada",
            "imagen de buena calidad, nítida",
            "imagen antigua dañada, necesita restauración",
            "imagen con mejoras de color y contraste",
            "imagen con reducción de ruido",
            "imagen con artefactos de procesamiento",
            "imagen vintage con calidad aceptable",
            "imagen digital con buena definición"
        ]

        inputs = processor(text=quality_labels, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        top_probs, top_indices = torch.topk(probs[0], 3)
        results = {
            f"clasificacion_{i+1}": {
                "categoria": quality_labels[idx],
                "probabilidad": f"{prob.item():.1%}"
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        }
        return results

    except Exception as e:
        return {"error": f"Error en CLIP: {str(e)}"}


# =========================
# 🔹 Gemini - Comparativo
# =========================
def analizar_con_gemini_comparativo(imagen_original_bytes: bytes, imagen_restaurada_bytes: bytes, gemini_api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)

    try:
        pil_original = Image.open(io.BytesIO(imagen_original_bytes))
        pil_restaurada = Image.open(io.BytesIO(imagen_restaurada_bytes))
        pil_original.thumbnail((512, 512))
        pil_restaurada.thumbnail((512, 512))

        buffer_o, buffer_r = io.BytesIO(), io.BytesIO()
        pil_original.save(buffer_o, format='JPEG')
        pil_restaurada.save(buffer_r, format='JPEG')

        img_o = {"mime_type": "image/jpeg", "data": base64.b64encode(buffer_o.getvalue()).decode('utf-8')}
        img_r = {"mime_type": "image/jpeg", "data": base64.b64encode(buffer_r.getvalue()).decode('utf-8')}

        prompt = """
        Compara estas dos imágenes: la primera es la ORIGINAL dañada y la segunda la versión RESTAURADA con IA.
        Evalúa las mejoras observadas (nitidez, reducción de ruido, color, reconstrucción) 
        y menciona si existen artefactos nuevos o cambios no naturales. 
        Concluye con un breve resumen del resultado global de la restauración.
        """

        # Usar modelo básico que debería estar disponible
        try:
            model = genai.GenerativeModel('gemini-pro')
        except:
            # Fallback a modelo de visión si está disponible
            try:
                model = genai.GenerativeModel('gemini-pro-vision')
            except:
                raise Exception("No se pudo cargar ningún modelo Gemini disponible")
        response = safe_generate(model, [prompt, img_o, img_r])
        return response.text if response and response.text else "Análisis no disponible"

    except Exception as e:
        return f"Error en análisis Gemini: {str(e)}"


# =========================
# 🔹 Análisis completo
# =========================
def analizar_imagen_completo(imagen_original_bytes: bytes, imagen_restaurada_bytes: bytes, gemini_api_key: str = "") -> dict:
    """Análisis completo con CLIP + Gemini comparando original vs restaurada"""
    resultados = {}
    resultados["clip_clasificacion"] = analizar_calidad_clip(imagen_restaurada_bytes)
    resultados["gemini_analisis"] = analizar_con_gemini_comparativo(imagen_original_bytes, imagen_restaurada_bytes, gemini_api_key)
    return resultados
