# 🎨 Segmentación Simple por Color en Imágenes

¡Bienvenido a este laboratorio introductorio a la segmentación de imágenes! 🚀

Este repositorio contiene un notebook de Jupyter (`.ipynb`) que te enseñará una de las técnicas más intuitivas y fundamentales para aislar objetos en una imagen: la **segmentación por color**. Aprenderás a identificar y separar objetos basándote en un rango de colores específico.

## 🧠 ¿Qué aprenderás aquí?

A través de un ejemplo práctico con una imagen de flores, este notebook te guiará paso a paso para:

* **Analizar Imágenes:** Cargarás una imagen y extraerás sus propiedades básicas como tamaño y rango de valores de píxeles.
* **Separar Canales de Color:** Descompondrás la imagen en sus canales individuales (en este caso, BGR) para analizar la contribución de cada color.
* **Crear Máscaras de Color:** Implementarás un umbral (threshold) para crear una máscara binaria que identifique los píxeles que caen dentro de un rango de color deseado (por ejemplo, el rojo de las flores).
* **Limpiar la Máscara:** Aplicarás operaciones morfológicas, como la **apertura**, para eliminar el ruido y las pequeñas imperfecciones de la máscara.
* **Encontrar el Objeto:** Utilizarás la máscara limpia para calcular un **rectángulo delimitador** (`bounding box`) que encierre el objeto de interés.
* **Detectar Bordes:** Implementarás un algoritmo simple para detectar los bordes del objeto segmentado.

## 🚀 Cómo Empezar

1.  **Clona o descarga** este repositorio.
2.  **Abre el notebook** `06_Segmentacion_Simple_Color.ipynb` en un entorno como [Google Colab](https://colab.research.google.com/).
3.  **Ejecuta las celdas** en orden para seguir todo el proceso de segmentación, desde la carga de la imagen hasta la visualización de los resultados finales.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **OpenCV (`cv2`)**
* **NumPy**
* **Matplotlib**

## ✅ Conclusión Clave

La segmentación por color es una técnica poderosa y eficiente cuando los objetos de interés tienen un color distintivo que los diferencia del fondo. Es un primer paso excelente para tareas más complejas como el reconocimiento, seguimiento o conteo de objetos.