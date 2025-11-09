import os
import io
import base64
import google.generativeai as genai

def list_models(api_key):
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        print("Modelos disponibles:")
        for m in models:
            print(" -", m.name, "| métodos:", m.supported_generation_methods)
    except Exception as e:
        print("Error listando modelos:", e)

def test_generate(api_key, model_name):
    genai.configure(api_key=api_key)
    prompt = "Escribe una frase breve de prueba."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        print("Respuesta del modelo:", response.text)
    except Exception as e:
        print(f"Error usando el modelo {model_name}:", e)

if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ No se encontró la variable de entorno GEMINI_API_KEY")
    else:
        list_models(key)
        # Cambiá aquí por los modelos que viste en la lista
        test_generate(key, "gemini-2.5-flash")
        test_generate(key, "gemini-2.5-pro")
