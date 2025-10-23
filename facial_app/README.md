#  Detector de Landmarks Faciales 👁️👄

Aplicación web interactiva para detectar 478 puntos clave (landmarks) en rostros humanos, construida con MediaPipe y Streamlit.

---

## 🚀 Características Principales

-   **Detección de Alta Precisión:** Identifica 478 landmarks faciales usando el modelo Face Mesh de MediaPipe.
-   **Interfaz Web Interactiva:** Permite a los usuarios subir sus propias imágenes para analizarlas.
-   **Visualización Clara:** Muestra una comparación lado a lado de la imagen original y la imagen con los landmarks dibujados.

---

## 🛠️ Tecnologías Utilizadas

-   **Python 3.11+**
-   **MediaPipe**: Para el modelo de detección de landmarks.
-   **OpenCV**: Para el procesamiento de imágenes (lectura, dibujo de círculos).
-   **Streamlit**: Para la creación de la interfaz web.

---

## 📦 Instalación y Ejecución Local

Sigue estos pasos para ejecutar la aplicación en tu máquina local.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Nulltheory/Tecnicas_de_Procesamiento_de_Imagenes/facial-landmarks-app.git](https://github.com/Nulltheory/Tecnicas_de_Procesamiento_de_Imagenes/facial-landmarks-app.git)
    cd facial-landmarks-app
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    # Crear el entorno
    python -m venv venv
    
    # Activar en Windows (CMD/PowerShell)
    .\venv\Scripts\activate
    
    # Activar en Linux/Mac (Bash/Zsh)
    source venv/bin/activate
    ```

3.  **Instalar las dependencias:**
    Asegúrate de tener tu archivo `requirements.txt` en la carpeta.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación Streamlit:**
    ```bash
    streamlit run app.py
    ```
    La aplicación se abrirá automáticamente en tu navegador web.

---

## ☁️ Deployment en Streamlit Community Cloud

Puedes hostear esta aplicación gratis siguiendo estos pasos:

1.  Sube todo tu código (incluyendo `app.py`, `requirements.txt` y tu `.gitignore`) a un repositorio público en GitHub.
2.  Ve a [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión.
3.  Haz clic en "**New app**" y conecta tu repositorio de GitHub.
4.  Asegúrate de que la rama (`main` o `master`) y el archivo principal (`app.py`) estén correctos.
5.  ¡Haz clic en "**Deploy!**".

---

## 📚 Documentación y Recursos

-   [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
-   [Documentación Oficial de Streamlit](https://docs.streamlit.io/)
-   Kilo Code

## 🎓 Autor

Desarrollado como parte del **Laboratorio 2** de la materia **Procesamiento Digital de Imágenes** (IFTS24).

---

## 📜 Licencia

Este proyecto está bajo la [Licencia MIT](https://choosealicense.com/licenses/mit/).