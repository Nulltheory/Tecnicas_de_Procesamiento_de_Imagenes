import streamlit as st
import google.generativeai as genai

st.title("Prueba Google Gemini API")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Hola Gemini, ¿me recibes?")

    st.success("✅ Conectado correctamente")
    st.write("**Respuesta del modelo:**")
    st.write(response.text)

except Exception as e:
    st.error(f"❌ Error al probar Gemini: {e}")
