# 🎨 Espacios de Color: El Dilema entre RGB y BGR

¡Bienvenido a un tutorial clave en el mundo del procesamiento de imágenes! 🚀 Este repositorio contiene un notebook de Jupyter (`.ipynb`) que aborda uno de los problemas más comunes y confusos al empezar: la diferencia entre los espacios de color **RGB** y **BGR**.

Entender esta diferencia es fundamental para evitar errores de visualización y asegurar que tus algoritmos procesen los colores correctamente.

## 🧠 ¿Qué aprenderás aquí?

Este notebook te enseñará de manera práctica y visual a:

* **Identificar** los dos espacios de color más comunes:
    * **RGB (Red, Green, Blue):** El estándar para la mayoría de las pantallas y librerías de visualización como Matplotlib.
    * **BGR (Blue, Green, Red):** El estándar adoptado por OpenCV, una de las librerías de visión por computadora más potentes.
* **Cargar imágenes** utilizando OpenCV y entender por qué los colores pueden parecer "incorrectos" al visualizarlas con otras herramientas.
* **Separar una imagen** en sus canales de color individuales (R, G, B) y analizarlos.
* **Convertir imágenes** de manera eficiente entre BGR y RGB para asegurar la compatibilidad entre diferentes librerías.
* **Visualizar comparativamente** los resultados para entender el impacto de una conversión incorrecta frente a una correcta.

## 🚀 Cómo Empezar

1.  **Clona o descarga** este repositorio.
2.  **Abre el notebook** `02_Espacios_Color_RGB_BGR.ipynb` en un entorno como [Google Colab](https://colab.research.google.com/) o Jupyter.
3.  **Ejecuta las celdas** en orden para seguir la explicación paso a paso y ver cómo los colores cambian con cada operación.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **OpenCV (`cv2`)**
* **NumPy**
* **Matplotlib**

## ✅ Conclusión Clave

La regla de oro: **si cargas una imagen con OpenCV y la quieres visualizar con Matplotlib, ¡siempre conviértela de BGR a RGB primero!**