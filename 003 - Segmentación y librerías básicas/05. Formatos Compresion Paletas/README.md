# 💾 Formatos, Compresión y Paletas de Color en Imágenes

¡Bienvenido a este laboratorio práctico sobre los formatos de imagen y las técnicas de compresión! 🚀

Este repositorio contiene un notebook de Jupyter (`.ipynb`) que explora cómo se guardan, comprimen y representan las imágenes digitales. Entender estos conceptos es crucial para optimizar el almacenamiento, la transmisión y la calidad de los datos visuales en cualquier aplicación.

## 🧠 ¿Qué aprenderás aquí?

En este notebook, desmitificaremos los conceptos clave que determinan cómo se manejan los archivos de imagen:

* **Formatos Populares:** Aprenderás las diferencias fundamentales, ventajas y desventajas de los formatos más comunes:
    * **JPG/JPEG:** Ideal para fotografías, utiliza compresión con pérdida.
    * **PNG:** Perfecto para gráficos con transparencias, utiliza compresión sin pérdida.
    * **GIF:** Famoso por las animaciones, utiliza una paleta de colores limitada.

* **Compresión de Datos:** Entenderás los dos tipos principales de compresión:
    * **Con Pérdida (Lossy):** Reduce drásticamente el tamaño del archivo sacrificando algo de calidad (ej. JPG).
    * **Sin Pérdida (Lossless):** Reduce el tamaño del archivo sin perder ni un solo bit de información (ej. PNG).

* **Paletas de Color (Imágenes Indexadas):** Descubrirás cómo formatos como GIF reducen el tamaño del archivo al limitar la imagen a una paleta de colores predefinida (normalmente 256 colores), en lugar de almacenar el valor de cada píxel individualmente.

## 🚀 Cómo Empezar

1.  **Clona o descarga** este repositorio.
2.  **Abre el notebook** `05_Formatos_Compresion_Paletas.ipynb` en un entorno como [Google Colab](https://colab.research.google.com/) o Jupyter.
3.  **Ejecuta las celdas** en orden para ver los ejemplos prácticos de cómo guardar imágenes en diferentes formatos y con distintos niveles de calidad.

## 🛠️ Tecnologías Utilizadas

* **Python 3**
* **Pillow (PIL)**
* **Matplotlib**
* **Scikit-image**

## ✅ Conclusión Clave

La elección del formato y el nivel de compresión es una decisión estratégica que depende del caso de uso. No hay un "mejor" formato, sino el formato "más adecuado" para cada situación, balanceando siempre la calidad de la imagen con el tamaño del archivo.