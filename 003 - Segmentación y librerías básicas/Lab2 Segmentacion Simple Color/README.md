# 🎨 Laboratorio 2: Segmentación Simple por Color

¡Bienvenido a esta aplicación práctica del Procesamiento Digital de Imágenes! 🚀

Este repositorio contiene un notebook de Jupyter (`.ipynb`) que te enseñará a resolver un problema real y fundamental en la visión por computadora: la **segmentación de imágenes**. El objetivo es aplicar los conceptos que ya conoces (espacios de color, cuantización, muestreo) para dividir una imagen en regiones significativas.

## 🧠 ¿Qué aprenderás aquí?

Este laboratorio te guiará para que domines el flujo de trabajo completo de la segmentación por color, una técnica esencial en muchísimos campos:

* **¿Qué es la Segmentación?** Entenderás por qué separar los objetos del fondo es un paso crucial en aplicaciones que van desde el diagnóstico médico hasta el control de calidad industrial.
* **Análisis de Histogramas:** Aprenderás a usar histogramas como una herramienta para analizar la distribución de colores y elegir los umbrales (límites) correctos para tu segmentación.
* **Creación de Máscaras:** Implementarás un filtro para crear una máscara binaria, que es el corazón de la segmentación, para aislar los píxeles que corresponden a tu objeto de interés.
* **Limpieza y Refinamiento:** Utilizarás operaciones morfológicas, como la **apertura**, para eliminar el ruido y perfeccionar la forma del objeto detectado.
* **Extracción de Características:** Aprenderás a usar la máscara final para "recortar" el objeto de la imagen original y analizarlo de forma aislada.

## 🚀 Cómo Empezar

1.  **Clona o descarga** este repositorio.
2.  **Abre el notebook** `Lab2 Segmentacion Simple Color.ipynb` en un entorno como [Google Colab](https://colab.research.google.com/).
3.  **Ejecuta las celdas** en orden para seguir el proceso completo, desde el análisis inicial hasta la extracción final del objeto.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **NumPy**
* **Matplotlib**
* **Scikit-image**

## ✅ Conexión de Conceptos

Este laboratorio conecta todo lo aprendido:
* **Espacios de Color:** La elección entre RGB y HSV afecta directamente la facilidad de la segmentación.
* **Cuantización y Muestreo:** Determinan la precisión y resolución de nuestros resultados.
* **Análisis Estadístico:** Los histogramas son nuestra guía para tomar decisiones basadas en datos.