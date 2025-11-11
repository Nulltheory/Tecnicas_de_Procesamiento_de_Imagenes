---
title: Clasificador de Imagenes de Residuos Reciclables ♻️
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ♻️ Clasificador de Residuos Reciclables con IA

> 🧠 Proyecto desarrollado para la materia  
> **Procesamiento Digital de Imágenes y Visión por Computadora – 2025**

---

## 📖 Descripción

Esta aplicación utiliza **Inteligencia Artificial** para **clasificar imágenes de residuos reciclables** en distintas categorías como **plástico, vidrio, papel, cartón, metal u orgánico**.  
El objetivo es fomentar la **educación ambiental** y facilitar la **separación de residuos** mediante el uso de modelos de **visión por computadora**.

---

## 🧩 Modelos Utilizados

### 🔹 Modelo Preentrenado (CLIP)
- `openai/clip-vit-base-patch32`
- Clasificación *Zero-Shot* (sin entrenamiento previo)
- Basado en **Hugging Face Transformers**

### 🔹 Modelo Personalizado (Teachable Machine)
- Entrenado con dataset propio (~500 imágenes)
- Exportado en formato `.h5`
- Integrado con **TensorFlow** y **Gradio**

---

## 🗂️ Categorías

| Emoji | Categoría |
|:------:|:-----------|
| 🧴 | Plástico |
| 🍾 | Vidrio |
| 📄 | Papel |
| 📦 | Cartón |
| 🥬 | Orgánico |
| 🥫 | Metal |

---

## 🚀 Cómo usar

1. Subí una imagen o usá la cámara integrada.  
2. Presioná **“Clasificar con Modelo Personalizado”** o **“Clasificar con CLIP”**.  
3. Observá las predicciones y compará los resultados.  
4. Probá diferentes tipos de residuos para ver el comportamiento de ambos modelos.

---

## 🧪 Ejemplos de uso

| Imagen | Resultado esperado |
|--------|--------------------|
| 🧴 Botella de gaseosa | Plástico |
| 🍾 Frasco de mermelada | Vidrio |
| 📄 Hoja de cuaderno | Papel |
| 📦 Caja de cereales | Cartón |
| 🥬 Cáscara de banana | Orgánico |
| 🥫 Lata de gaseosa | Metal |

---

## ⚙️ Instalación local

```bash
# Clonar el repositorio
git clone [url-del-repositorio]

# Entrar al directorio
cd mi_proyecto_vision

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

---

## 📊 Comparación de Modelos

### 🧠 Modelo Preentrenado (CLIP)

**Ventajas**:
- No requiere entrenamiento
- Funciona con cualquier categoría en lenguaje natural
- Generaliza bien a diferentes contextos

**Desventajas**:
- Menor precisión en tareas muy específicas.
- No se adapta al dominio particular del dataset (por ejemplo, materiales con apariencia similar).
- Su inferencia puede ser más lenta o demandante en recursos.

**Resultados en mi dataset**:
- Precisión aproximada: 70%
- Casos donde funciona bien: distingue correctamente entre materiales visualmente diferentes (por ejemplo, *metal* vs. *papel*)
- Casos donde falla: tiende a confundir elementos de colores similares, como *desecho orgánico* y *vidrio*, que pueden ser de color *verde* o *marrón*. También confunde objetos que se encuentra agrupados como "montones" 


### ⚙️ Modelo Personalizado (Teachable Machine)

**Ventajas**:
- Alta precisión en la tarea específica
- Adaptado al dominio de interés
- Distingue correctamente materiales donde un mismo elemento predomina en la imagen.

**Desventajas**:
- Requiere recolectar y etiquetar datos
- Solo funciona para las clases entrenadas
- Puede sufrir overfitting con pocos datos
- Tiende a confundir elementos de colores similares, como *desecho orgánico* y *vidrio*, que pueden ser de color *verde* o *marrón*. También confunde objetos que se encuentra agrupados como "montones" 

**Resultados en mi dataset**:
- Precisión aproximada: +90%
- Tamaño del dataset de entrenamiento: +500 imágenes distribuidas en seis categorías.
- Mejora respecto a CLIP: Mayor precisión promedio sobre el mismo conjunto de validación.

---

## 🧩 Conclusiones

- El modelo **preentrenado (CLIP)** es una excelente opción para tareas exploratorias o cuando no se dispone de un dataset propio.  
  Sin embargo, su rendimiento se ve limitado en dominios muy específicos.

- El **modelo personalizado (Teachable Machine)**, al estar entrenado con ejemplos reales del dominio, logra un desempeño con mayor precisión, especialmente en contextos de aplicación concreta.

- Durante el desarrollo se aprendió sobre:
  - Preprocesamiento y normalización de imágenes.
  - Compatibilidad entre versiones de TensorFlow y Keras.
  - Integración de modelos en interfaces interactivas con Gradio.
  - Comparación práctica entre modelos *generalistas* y *especializados*.

- **Mejoras futuras:**
  - Ampliar el dataset con más ejemplos y condiciones de iluminación variadas.
  - Aplicar *data augmentation* para mejorar la robustez.
  - Implementar una arquitectura híbrida donde CLIP sirva como detector general y el modelo personalizado refine la predicción final.

- **Aplicaciones potenciales:**
  - Sistemas de reciclaje inteligente.
  - Educación ambiental interactiva.
  - Clasificación automatizada de materiales en plantas de tratamiento.

---

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License**.  
Podés usar, modificar y distribuir el código libremente, siempre que se mantenga el aviso de autoría original.

---

## 🙌 Créditos

**Autor**: Alejandro Schiariti  
**Institución**: Instituto de Formación Técnica Superior N°24
**Materia**: Procesamiento Digital de Imágenes y Visión por Computadora (2025)
**Año**: 2025  
**Tecnologías**: TensorFlow · Gradio · Hugging Face · Teachable Machine

---