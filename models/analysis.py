"""
🤖 Módulo de Análisis con IA

Este módulo proporciona funciones de análisis inteligente para evaluación
de calidad de imágenes usando modelos de IA avanzados.

Funcionalidades:
- Análisis comparativo con Gemini AI
- Clasificación automática con CLIP
- Detección de cambios y mejoras
- Análisis de calidad técnica

Modelos utilizados:
- Google Gemini 1.5 Pro/Flash
- OpenAI CLIP ViT-Base
- BLIP (opcional)
"""

import io
import base64
import os
import numpy as np
from PIL import Image

# --- BLIP HF ---
def analizar_con_blip_hf(hf_token: str, imagen_bytes: bytes) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model="Salesforce/blip-image-captioning-large", token=hf_token)
    result = client.image_to_text(imagen_bytes)
    return result[0].get("generated_text", "No se generó texto") if result else "No se generó texto"

# --- Gemini 2.0 ---
def analizar_con_gemini(gemini_api_key: str, imagen_bytes: bytes) -> str:
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    prompt = (
        "Eres un asistente experto en restauración de fotografías antiguas. "
        "Describe brevemente (2-4 frases) qué mejoras observas en esta imagen restaurada "
        "en términos de: rostros (nitidez, reconstrucción), fondo (ruido, artefactos), "
        "color/tonalidad y artefactos restantes. "
        "Indica también si la restauración introdujo cambios no naturales."
    )
def analizar_calidad_clip(imagen_bytes: bytes) -> dict:
    """Clasifica calidad de imagen usando CLIP"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        # Verificar que torch esté disponible y funcione
        if not torch.cuda.is_available() and not hasattr(torch, 'cpu'):
            return {"error": "PyTorch no está disponible"}

        # Cargar modelo CLIP con manejo de errores mejorado
        try:
            # Usar modelo alternativo que no requiere autenticación
            model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", local_files_only=False)
            processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", local_files_only=False)
        except Exception as model_error:
            # Fallback a un modelo aún más simple si el anterior falla
            try:
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
            except Exception as fallback_error:
                return {"error": f"Error cargando modelo CLIP: {str(model_error)}. Modelo alternativo también falló: {str(fallback_error)}"}

        # Convertir bytes a PIL Image
        try:
            image = Image.open(io.BytesIO(imagen_bytes))
            # Verificar que la imagen se cargó correctamente
            if image.size[0] == 0 or image.size[1] == 0:
                return {"error": "Imagen inválida o corrupta"}
        except Exception as img_error:
            return {"error": f"Error procesando imagen: {str(img_error)}"}

        # Categorías de calidad más descriptivas y útiles
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

        # Procesar con CLIP
        try:
            inputs = processor(text=quality_labels, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Obtener las 3 mejores clasificaciones
            top_probs, top_indices = torch.topk(probs[0], 3)

            results = {}
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                results[f"clasificacion_{i+1}"] = {
                    "categoria": quality_labels[idx],
                    "probabilidad": f"{prob.item():.1%}"
                }

            return results

        except Exception as proc_error:
            return {"error": f"Error en procesamiento CLIP: {str(proc_error)}"}

    except ImportError as ie:
        return {"error": f"Librerías faltantes: {str(ie)}. Instala transformers y torch"}
    except Exception as e:
        return {"error": f"Error general en CLIP: {str(e)}"}

def analizar_con_gemini_comparativo(imagen_original_bytes: bytes, imagen_restaurada_bytes: bytes, gemini_api_key: str) -> str:
    """Análisis comparativo detallado de calidad con Gemini entre original y restaurada"""
    if not gemini_api_key:
        return "API key de Gemini no configurada. Configure la clave en el panel lateral."

    # Verificar si las imágenes son realmente diferentes antes de enviar a Gemini
    try:
        pil_original = Image.open(io.BytesIO(imagen_original_bytes))
        pil_restaurada = Image.open(io.BytesIO(imagen_restaurada_bytes))

        import numpy as np
        arr_original = np.array(pil_original)
        arr_restaurada = np.array(pil_restaurada)

        if arr_original.shape == arr_restaurada.shape:
            diff = np.abs(arr_original.astype(np.int32) - arr_restaurada.astype(np.int32))
            max_diff = np.max(diff)

            if max_diff == 0:
                return """**⚠️ ANÁLISIS CANCELADO**: Las imágenes son idénticas (diferencia máxima = 0).

**Diagnóstico del problema:**
- Los modelos de restauración no aplicaron cambios significativos
- Es posible que los modelos fallaran silenciosamente
- Verifica que todos los pasos de procesamiento se ejecutaron correctamente
- Revisa los logs de error para identificar fallos en CodeFormer, GFPGAN o Real-ESRGAN

**Recomendación:** Ejecuta la restauración nuevamente y verifica que cada paso muestre "✅ [Modelo] aplicado exitosamente"."""

    except Exception as e:
        return f"Error verificando diferencias de imagen: {str(e)}"

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)

        # List available models first to find the correct ones
        try:
            available_models = genai.list_models()
            vision_models = [model for model in available_models if 'vision' in model.name.lower() or 'generateContent' in model.supported_generation_methods]
            model_names = [model.name for model in vision_models]
            print(f"Available vision models: {model_names}")
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            model_names = []

        # Try available models or fallback to common ones
        models_to_try = model_names if model_names else ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"]

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)

                # Convert both images to base64
                pil_original = Image.open(io.BytesIO(imagen_original_bytes))
                buffer_original = io.BytesIO()
                pil_original.save(buffer_original, format='JPEG')
                original_bytes_clean = buffer_original.getvalue()

                pil_restaurada = Image.open(io.BytesIO(imagen_restaurada_bytes))
                buffer_restaurada = io.BytesIO()
                pil_restaurada.save(buffer_restaurada, format='JPEG')
                restaurada_bytes_clean = buffer_restaurada.getvalue()

                # Create image parts for both images
                image_part_original = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(original_bytes_clean).decode('utf-8')
                }

                image_part_restaurada = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(restaurada_bytes_clean).decode('utf-8')
                }

                prompt = """
                Compara estas dos imágenes: la primera es la imagen ORIGINAL dañada, la segunda es la versión RESTAURADA con IA.

                IMPORTANTE: Evalúa las mejoras de manera POSITIVA Y CONSTRUCTIVA, enfocándote en lo que SÍ mejoró, aunque sea sutil.

                Proporciona un análisis OPTIMISTA Y ÚTIL de calidad:

                1. **Mejoras Observadas**: Destaca cualquier mejora, por pequeña que sea (mejor contraste, reducción de ruido, nitidez, colores)
                2. **Aspectos Técnicos Mejorados**: Menciona mejoras en calidad técnica (contraste, saturación, reducción de ruido digital)
                3. **Limitaciones Esperadas**: Menciona brevemente qué NO se pudo arreglar, pero enfócate en el progreso logrado
                4. **Recomendaciones**: Sugiere próximos pasos o ajustes para mejorar aún más
                5. **Resultado General**: Conclusión positiva sobre el valor de la restauración aplicada

                Formato: Responde en español con viñetas claras. Enfócate en lo POSITIVO y ÚTIL para el usuario.
                """

                response = model.generate_content([prompt, image_part_original, image_part_restaurada])
                return response.text if response and response.text else "Análisis no disponible - respuesta vacía del modelo"
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                continue

        return "No se pudo generar análisis con Gemini - todos los modelos fallaron. Verifique la API key y conexión."

    except Exception as e:
        return f"Error en análisis Gemini: {str(e)}"

def analizar_con_gemini_calidad(imagen_bytes: bytes, gemini_api_key: str) -> str:
    """Análisis detallado de calidad con Gemini"""
    if not gemini_api_key:
        return "API key de Gemini no configurada. Configure la clave en el panel lateral."

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)

        # List available models first to find the correct ones
        try:
            available_models = genai.list_models()
            vision_models = [model for model in available_models if 'vision' in model.name.lower() or 'generateContent' in model.supported_generation_methods]
            model_names = [model.name for model in vision_models]
            print(f"Available vision models: {model_names}")
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            model_names = []

        # Try available models or fallback to common ones
        models_to_try = model_names if model_names else ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"]

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)

                # Convert bytes to PIL Image and then back to bytes for proper format
                pil_image = Image.open(io.BytesIO(imagen_bytes))
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                image_bytes_clean = buffer.getvalue()

                # Use the image directly in generate_content instead of upload_file
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes_clean).decode('utf-8')
                }

                prompt = """
                Analiza esta imagen restaurada y proporciona un análisis detallado de calidad:

                1. **Calidad General**: Clasifica como Excelente/Buena/Regular/Mala
                2. **Nitidez**: Evalúa la nitidez general (Alta/Media/Baja)
                3. **Ruido**: Nivel de ruido presente (Bajo/Medio/Alto)
                4. **Colores**: Calidad de color y tonalidad (Naturales/Desvaídos/Mejorados)
                5. **Artefactos**: Presencia de artefactos de procesamiento (Ninguno/Pocos/Muchos)
                6. **Mejoras Observadas**: Qué aspectos mejoraron significativamente

                Formato: Responde en español con viñetas claras y concisas.
                """

                response = model.generate_content([prompt, image_part])
                return response.text if response and response.text else "Análisis no disponible - respuesta vacía del modelo"
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                continue

        return "No se pudo generar análisis con Gemini - todos los modelos fallaron. Verifique la API key y conexión."

    except Exception as e:
        return f"Error en análisis Gemini: {str(e)}"

def analizar_imagen_completo(imagen_original_bytes: bytes, imagen_restaurada_bytes: bytes, gemini_api_key: str = "") -> dict:
    """Análisis completo con CLIP + Gemini comparando original vs restaurada"""
    resultados = {}

    # Análisis CLIP de la imagen restaurada
    clip_results = analizar_calidad_clip(imagen_restaurada_bytes)
    resultados["clip_clasificacion"] = clip_results

    # Análisis Gemini comparativo
    gemini_analysis = analizar_con_gemini_comparativo(imagen_original_bytes, imagen_restaurada_bytes, gemini_api_key)
    resultados["gemini_analisis"] = gemini_analysis

    return resultados
