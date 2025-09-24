# 🎨 Operaciones Gráficas con OpenCV en Python 🐍

¡Bienvenido a este tutorial práctico sobre operaciones gráficas en imágenes! 🚀

En este repositorio encontrarás un notebook de Jupyter (`.ipynb`) diseñado para enseñarte a **dibujar figuras geométricas y añadir texto** sobre imágenes utilizando la potente librería **OpenCV**.

Este es un punto de partida fundamental para tareas de visión por computadora como la anotación de datos, la visualización de resultados o la creación de interfaces gráficas simples.

## 🖌️ ¿Qué aprenderás a hacer?

A través de ejemplos claros y concisos, este notebook te guiará para:

* 📏 **Dibujar Líneas:** Aprende a trazar líneas rectas con control total sobre el punto de inicio, fin, color y grosor.
* 🔵 **Crear Círculos:** Dibuja círculos perfectos, ya sean vacíos o completamente rellenos.
* 🖼️ **Trazar Rectángulos:** Define áreas de interés o resalta objetos dibujando rectángulos.
* ✍️ **Escribir Texto:** Agrega texto personalizado a tus imágenes, eligiendo la fuente, el tamaño, el color y la posición.

## 🚀 Cómo Empezar

Es muy sencillo poner en práctica este código.

1.  **Clona o descarga** este repositorio en tu máquina.
2.  **Abre el notebook** `08_Operaciones_Graficas.ipynb` en [Google Colab](https://colab.research.google.com/) o en tu entorno local de Jupyter.
3.  **Ejecuta las celdas** en orden para seguir el tutorial y ver los resultados en tiempo real. ¡No dudes en experimentar cambiando los parámetros!

## ✨ Ejemplo Rápido

Aquí tienes una pequeña muestra de lo que puedes lograr con el código de este tutorial:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creamos un lienzo en negro
imagen = np.zeros((512, 512, 3), dtype="uint8")

# Dibujamos un rectángulo azul
cv2.rectangle(imagen, (100, 100), (400, 250), (255, 0, 0), 3)

# Escribimos un texto en blanco
cv2.putText(imagen, "Hola OpenCV!", (110, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Mostramos la imagen
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Mi Creacion con OpenCV")
plt.show()