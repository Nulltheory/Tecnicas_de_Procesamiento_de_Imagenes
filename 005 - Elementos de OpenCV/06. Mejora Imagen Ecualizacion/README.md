# 🎨 Mejora de Color en Imágenes con OpenCV

¡Bienvenido a este laboratorio práctico sobre mejora de la luminosidad y el contraste en imágenes! 🚀

Este repositorio contiene un notebook de Google Colab (`.ipynb`) que te guiará a través de varias técnicas fundamentales para realzar la calidad visual de una imagen. Si alguna vez has trabajado con imágenes oscuras o con poco contraste, ¡aquí encontrarás las herramientas para solucionarlo!

## ✨ ¿Qué aprenderás en este proyecto?

A través de ejemplos prácticos y visuales, exploraremos cómo:

* 📊 **Analizar Histogramas:** Entenderemos qué es un histograma y cómo nos revela información crucial sobre el brillo y contraste de una imagen.
* ⚖️ **Ecualización de Histograma:** Aplicaremos la ecualización estándar para mejorar el contraste de manera global en imágenes en escala de grises.
* 🌈 **Ecualización en Color (HSV y Lab):** Descubriremos por qué no se debe ecualizar directamente en el espacio de color RGB y aprenderemos a hacerlo correctamente trabajando con los canales de luminancia en los espacios **HSV** y **Lab**.
* 💡 **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Utilizaremos esta técnica avanzada para mejorar el contraste de forma local, obteniendo resultados mucho más naturales y evitando el ruido excesivo.

## 🚀 Cómo Empezar

¡Es muy fácil! Sigue estos pasos para ejecutar el código:

1.  **Abre el notebook en Google Colab:** La forma más sencilla es cargar el archivo `06_Mejora_Imagen_Ecualizacion.ipynb` en [Google Colab](https://colab.research.google.com/).
2.  **Ejecuta las celdas:** Sigue las instrucciones y ejecuta el código celda por celda.
3.  **Experimenta por tu cuenta:** ¡No dudes en subir tus propias imágenes y aplicar estas técnicas para ver cómo mejoran!

## 🛠️ Herramientas Utilizadas

* **Python 3**
* **OpenCV**
* **NumPy**
* **Matplotlib**