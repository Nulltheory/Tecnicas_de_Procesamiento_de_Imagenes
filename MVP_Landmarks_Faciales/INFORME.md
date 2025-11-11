# 🛸 INFORME TÉCNICO: Detector de Landmarks Faciales

**Materia:** Procesamiento Digital de Imágenes
**Laboratorio:** 2 - Detección de Landmarks
**Autor:** Alejandro Schiariti

---

## 1. Introducción 🎯

Los **landmarks faciales** (o puntos clave faciales) son un conjunto de puntos localizados con precisión en las distintas regiones de un rostro 👱, como los ojos, las cejas, la nariz, la boca y el contorno de la cara.

Su importancia radica en que actúan como una "radiografía" estructural de la cara, convirtiendo una imagen no estructurada (píxeles) en un formato de datos estructurado (un grafo de coordenadas 🗺️). Esto permite a las computadoras "entender" la geometría y la topología del rostro.

Son la tecnología fundamental detrás de innumerables aplicaciones modernas, entre ellas:

* **Realidad Aumentada:** Filtros de Instagram o Snapchat que se ajustan perfectamente a la cara 👻.
* **Análisis de Expresiones:** Entender emociones (alegría, sorpresa) midiendo las distancias entre puntos (ej. comisuras de los labios 😃).
* **Autenticación Biométrica:** Reconocimiento facial avanzado 🔒.
* **Animación Facial:** "Mapear" los movimientos de la cara de un actor a un personaje digital (VFX, Animoji 👽).

Este proyecto implementa un detector de 478 landmarks faciales utilizando la biblioteca **MediaPipe** de Google, presentado a través de una aplicación web interactiva construida con **Streamlit**.

---

## 2. Arquitectura del Proyecto 🏗️

El proyecto se estructuró con una clara separación entre la interfaz de usuario y la lógica de procesamiento, facilitando el mantenimiento y la escalabilidad. La aplicación fue diseñada para ser desplegada en un entorno de contenedores (Docker) en Hugging Face Spaces.

La estructura de archivos principal es la siguiente:

```
facial-landmarks-app/
│
├── .streamlit/
│   └── config.toml         # Configuración de Streamlit (p.ej. para fix de CSRF)
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Constantes y configuración del modelo MediaPipe
│   ├── detector.py         # Clase 'FaceLandmarkDetector' (lógica de MediaPipe)
│   └── utils.py            # Funciones de ayuda (conversión PIL <-> CV2)
│
├── app.py                  # Script principal de Streamlit (la Interfaz de Usuario)
├── Dockerfile              # Instrucciones para construir el contenedor en HF
├── INFORME.md              # Este informe
├── README.md               # Instrucciones de uso e instalación
└── requirements.txt        # Dependencias de Python
```

---

## 3. Decisiones de Diseño 💡

1.  **Separación de Lógica (Clase `FaceLandmarkDetector`):** En lugar de escribir toda la lógica de MediaPipe dentro de `app.py`, se encapsuló en una clase `FaceLandmarkDetector` dentro de `src/detector.py`.
    * **Ventaja:** `app.py` se mantiene limpio y solo se ocupa de la UI (widgets, imágenes). Si quisiéramos cambiar MediaPipe por otra biblioteca, solo modificaríamos `detector.py` sin tocar la interfaz.

2.  **Caché de Streamlit (`@st.cache_resource` y `@st.cache_data`):** Esta fue la decisión de diseño más importante para la performance de la aplicación.
    * `@st.cache_resource` 💾: Se usó para `cargar_detector()`. Esto asegura que el pesado modelo de MediaPipe se cargue en memoria **una sola vez** cuando la app se inicia, y no en cada re-ejecución del script.
    * `@st.cache_data` ⚡️: Se usó para `procesar_imagen_cacheada()`. Esta función recibe los bytes de la imagen y el estilo, y guarda el resultado (los bytes de la imagen procesada y la info). Esto evita que la detección se re-calcule si el usuario solo cambia un *widget* que no afecta a la imagen.

3.  **Deployment con `Dockerfile` en Hugging Face:**
    * **Ventaja:** En lugar de depender de la configuración estándar de Streamlit Community Cloud, usar un `Dockerfile` nos dio control total sobre el entorno 🐳. Esto fue **esencial** para resolver los problemas de dependencias, como fijar la versión de Python a `3.11-slim` (en lugar de la `3.13` que fallaba) e instalar librerías de sistema (`apt-get install libgl1`) que OpenCV necesitaba.

4.  **Procesamiento de Bytes:** La función de caché y los widgets `st.image` fueron diseñados para operar directamente con `bytes` de imagen (en formato PNG).
    * **Ventaja:** Es más estable para el sistema de caché y, como descubrimos, fue una parte clave para solucionar el *bug* de la vibración, ya que `st.image` maneja `bytes` de forma más predecible que objetos PIL.

---

## 4. Desafíos Técnicos y Soluciones 🛠️

El desarrollo de este proyecto presentó varios desafíos significativos, principalmente relacionados con el entorno y el comportamiento de Streamlit.

### Desafío 1: `FileNotFoundError` del Modelo de MediaPipe 🚫

* **Problema:** Al ejecutar el proyecto localmente, la línea `mp.solutions.face_mesh.FaceMesh()` fallaba con un `FileNotFoundError`, no encontrando el archivo `.binarypb` del modelo.
* **Diagnóstico:** El problema no era el código, sino un conflicto de dependencias. `mediapipe==0.10.x` es incompatible con las versiones más nuevas de la biblioteca `protobuf` (`4.24.x+`) que se instalan por defecto.
* **Solución:** Forzar una versión compatible de `protobuf` en el `requirements.txt`.
    ```txt
    # requirements.txt
    mediapipe
    ...
    protobuf==3.20.3 
    ```

### Desafío 2: Falla de *Build* en Hugging Face 📦🔥

* **Problema:** El `Dockerfile` autogenerado por Hugging Face fallaba con `Job failed with exit code: 1` durante el paso `RUN pip3 install -r requirements.txt`.
* **Diagnóstico:** Hubo dos problemas:
    1.  La plantilla usaba `python:3.13-slim`, una versión demasiado nueva para la cual `mediapipe` y `opencv-python` no tenían binarios pre-compilados.
    2.  La imagen "slim" de Debian carecía de las librerías de sistema (C++) que OpenCV necesita para instalarse.
* **Solución:** Modificar el `Dockerfile`:
    1.  Cambiar la imagen base a `FROM python:3.11-slim`.
    2.  Añadir un comando `RUN apt-get install` para instalar las dependencias faltantes (`libgl1`, `libglib2.0-0`, etc.).

### Desafío 3: `AxiosError: Request failed with status code 403` 🛑

* **Problema:** La aplicación desplegada funcionaba, pero al intentar subir una imagen, la subida fallaba con un error 403 (Prohibido).
* **Diagnóstico:** Este error es causado por la protección CSRF (`enableXsrfProtection = true`) que Streamlit tiene activada por defecto, la cual entra en conflicto con el sistema de *proxy* que usa Hugging Face Spaces.
* **Solución:** Desactivar esta protección creando un archivo de configuración `.streamlit/config.toml` con el siguiente contenido:
    ```toml
    [server]
    enableXsrfProtection = false
    ```
    Y luego, asegurarse de copiar esta carpeta al contenedor usando `COPY .streamlit/ ./.streamlit/` en el `Dockerfile`.

### Desafío 4: El Bug de la "Vibración" (Bucle de Re-ejecución) 😵‍💫

* **Problema:** Tras procesar una imagen, la aplicación entraba en un bucle infinito de re-ejecución, causando un parpadeo o "vibración" constante.
* **Diagnóstico:** Este fue el desafío más complejo.
    1.  **Intento 1:** Usar `@st.cache_resource` para el modelo. Esto hizo la vibración *más rápida*, probando que el cuello de botella era el modelo, pero no el *disparador* del bucle.
    2.  **Intento 2:** Usar `st.session_state` y `st.button` para controlar el flujo. Esto tampoco funcionó, revelando que el bucle era causado por el renderizado en sí.
    3.  **Diagnóstico Final:** El verdadero culpable era `st.image(..., use_container_width=True)`. En el entorno del *iframe* de Hugging Face, la imagen se expandía, esto causaba un evento de "resize" en el *iframe*, lo cual a su vez le decía a Streamlit que la ventana había cambiado de tamaño, y Streamlit re-ejecutaba el script para "adaptarse" al nuevo tamaño, creando un bucle.
* **Solución:** Eliminar `use_container_width=True` de todas las llamadas a `st.image`. Al renderizar las imágenes con un tamaño fijo (basado en nuestro `resize_image` a 800px), el bucle de "resize" se rompió y la interfaz se estabilizó por completo.

---

## 5. Conclusiones y Aprendizajes 🎓

Este laboratorio demostró ser mucho más que un simple ejercicio de uso de una biblioteca de CV. Los aprendizajes principales fueron:

1.  **El Entorno es Clave:** Una aplicación funcional localmente puede fallar de formas inesperadas al ser desplegada. La depuración de entornos (versiones de Python, `protobuf`, librerías de sistema en Docker) es tan importante como la depuración del código Python.
2.  **Streamlit es Potente pero Tiene Peculiaridades:** El modelo de re-ejecución de Streamlit requiere un manejo cuidadoso del estado y el caché (`@st.cache_resource`, `@st.cache_data`).
3.  **La Causa Raíz No Siempre es Obvia:** El *bug* de la vibración fue un claro ejemplo de cómo una simple línea de configuración de UI (`use_container_width=True`) podía interactuar con el entorno de *deployment* (el *iframe* de HF) para crear un error de lógica de aplicación (el bucle de re-ejecución).
4.  **Diseño Modular:** Haber separado `detector.py` de `app.py` desde el inicio hizo que agregar las nuevas funcionalidades (Opción A, B, y D) fuera trivial, ya que solo tuvimos que modificar la interfaz y las funciones de procesamiento sin tocar el núcleo del detector.