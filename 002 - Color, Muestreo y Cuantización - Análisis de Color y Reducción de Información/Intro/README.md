# 💡 Normalizando la Iluminación en Imágenes 💡

¡Bienvenido! 🚀 Este repositorio contiene un notebook de Jupyter que demuestra una técnica específica y muy útil para el pre-procesamiento de imágenes: la **normalización de la iluminación por filas**.

## 🤔 ¿Cuál es el objetivo?

El objetivo de este código es simple pero potente: **reducir las variaciones de iluminación y mejorar el contraste local** en una imagen. Esto se logra procesando la imagen fila por fila para hacer que el fondo sea más uniforme y los objetos de interés resalten mucho más.

## 🧠 El Algoritmo: Paso a Paso

El notebook implementa un algoritmo claro y conciso para lograr la normalización:

1.  **Obtener Dimensiones:** Primero, identificamos el número de filas (N) y columnas (M) de la imagen de entrada.
2.  **Crear un Lienzo Vacío:** Se genera una nueva matriz de ceros (`Xm`) con las mismas dimensiones que la imagen original.
3.  **Procesar Fila por Fila:** Recorremos cada una de las filas de la imagen. En cada fila:
    * Se encuentra el valor de píxel **mínimo** (`xmin`) de esa fila en particular.
    * Se le **resta este valor mínimo** a todos los píxeles de esa misma fila.

Este proceso efectivamente "empuja" el valor más oscuro de cada fila a cero, realzando el rango dinámico del resto de los píxeles en esa línea.

## ✨ ¿Por qué es útil esta técnica?

Esta normalización es especialmente valiosa cuando necesitas:

* Reducir el efecto de sombras o gradientes de luz.
* Hacer que un fondo no uniforme se vuelva más consistente.
* Resaltar objetos para facilitar una posterior segmentación o umbralización (thresholding).
* Mejorar el contraste en imágenes de documentos o texto.

## 🚀 Cómo Empezar

1.  **Clona o descarga** este repositorio.
2.  **Abre el notebook** `Intro.ipynb` en un entorno como [Google Colab](https://colab.research.google.com/) o Jupyter.
3.  **Ejecuta las celdas** para ver el código en acción y el resultado de la transformación.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **NumPy**
* **Scikit-image**
* **Matplotlib**