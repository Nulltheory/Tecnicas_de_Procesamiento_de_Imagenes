---
title: Restaurador Fotográfico AI
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# 🧠 Restaurador Fotográfico AI

Transforma fotos antiguas dañadas con IA avanzada y obtén análisis detallado de mejoras. Una aplicación web completa para restauración fotográfica profesional con algoritmos de vanguardia.

## ✨ Características Principales

### 🎨 **9 Algoritmos de Restauración**
- **Real-ESRGAN**: Upscaling x4 de ultra-alta calidad
- **CodeFormer**: Restauración facial inteligente sin recortes
- **GFPGAN**: Restauración facial alternativa con recortes
- **CLAHE**: Contraste adaptativo para mejor definición local
- **Unsharp Mask**: Afilado sutil de detalles finos
- **Reducción de Ruido**: Eliminación avanzada de granulado
- **Reparación de Arañazos**: Detección y eliminación agresiva
- **Reparación de Manchas**: Eliminación de imperfecciones blancas
- **Stable Diffusion**: Mejoras creativas con IA generativa

### 🤖 **Análisis Inteligente con IA**
- **Gemini AI**: Análisis comparativo detallado original vs restaurado
- **CLIP Classification**: Análisis automático de contenido y calidad
- **Detección de Cambios**: Comparación pixel-perfect entre imágenes

### 🎛️ **Modos de Funcionamiento**
- **Modo Básico**: Configuración automática optimizada para principiantes
- **Modo Avanzado**: Control completo con presets inteligentes
- **Configuración Manual**: Control granular de todos los parámetros

## 🚀 Instalación y Uso

### 🌐 **Opciones de Despliegue Gratuito**

Esta aplicación requiere PyTorch y puede desplegarse en varias plataformas gratuitas:

#### 1️⃣ **Streamlit Cloud (Recomendado para PyTorch)**
- ✅ **Ir a** [Streamlit Cloud](https://streamlit.io/cloud)
- ✅ **Conectar con GitHub** y seleccionar el repositorio
- ✅ **Configurar secrets** en el panel de settings:
  - `GEMINI_API_KEY`: Para análisis con Gemini AI
  - `HF_TOKEN`: Para modelos avanzados de CLIP
- ✅ **¡Listo!** Despliegue automático optimizado para Streamlit + PyTorch

#### 2️⃣ **Hugging Face Spaces (Alternativa)**
- ✅ **Crear un Space** en [Hugging Face Spaces](https://huggingface.co/spaces)
- ✅ **Subir los archivos** del proyecto al repositorio
- ✅ **Configurar Secrets** (opcional):
  - `GEMINI_API_KEY`: Para análisis con Gemini AI
  - `HF_TOKEN`: Para modelos avanzados de CLIP
- ✅ **¡Listo!** La aplicación se desplegará automáticamente

#### 2️⃣ **Streamlit Cloud (Alternativa)**
- ✅ **Ir a** [Streamlit Cloud](https://streamlit.io/cloud)
- ✅ **Conectar con GitHub** y seleccionar el repositorio
- ✅ **Configurar secrets** en el panel de settings
- ✅ **Desplegar automáticamente**

#### 3️⃣ **Render (Otro alternativa)**
- ✅ **Crear cuenta** en [Render](https://render.com)
- ✅ **Conectar repositorio** de GitHub
- ✅ **Seleccionar "Web Service"** con Docker
- ✅ **Configurar variables de entorno**

#### 4️⃣ **Railway (Otra opción)**
- ✅ **Ir a** [Railway](https://railway.app)
- ✅ **Conectar con GitHub**
- ✅ **Configurar variables de entorno**
- ✅ **Desplegar con un click**

### ⚙️ **Configuración de Variables de Entorno**

Para funcionalidad completa, configurar estas variables:

```bash
GEMINI_API_KEY=tu-api-key-de-google
HF_TOKEN=tu-token-de-huggingface
```

### 💡 **Solución de Problemas con PyTorch**

**Si PyTorch no instala correctamente:**
1. **Verificar logs** de construcción de la plataforma
2. **Contactar soporte** de la plataforma utilizada
3. **Probar otra plataforma** de despliegue
4. **Usar GPU instances** si están disponibles

### 🎯 **Características del Despliegue**
- ✅ **PyTorch requerido** para funcionamiento completo
- ✅ **9 algoritmos de IA** para restauración avanzada
- ✅ **Modelos desde HF Hub** (sin descargas externas)
- ✅ **Procesamiento automático** GPU/CPU
- ✅ **Escalado automático** según demanda

### 💻 **Instalación Local (Desarrollo)**

#### 📋 Requisitos Previos
- Python 3.10+
- pip
- Conexión a internet (para descargar modelos)
- 4GB+ RAM recomendado
- GPU NVIDIA (opcional, mejora rendimiento)

#### 🛠️ Instalación Local

1. **Clona el repositorio:**
```bash
git clone <https://github.com/Nulltheory/>
cd restaurador-mvp
```

2. **Instala dependencias:**
```bash
pip install -r requirements.txt
```

3. **Configura APIs (Opcional pero recomendado):**
```bash
# Para análisis con Gemini
export GEMINI_API_KEY="tu-api-key-aqui"

# Para modelos avanzados de CLIP
export HF_TOKEN="tu-huggingface-token-aqui"
```

4. **Ejecuta la aplicación:**
```bash
streamlit run app.py
```

### 🌐 Acceso Web
Abre tu navegador en `http://localhost:8501`

### 🐳 Uso con Docker (Alternativo)
```bash
docker build -t restaurador-ai .
docker run -p 8501:8501 restaurador-ai
```

### 💻 Uso Local (Desarrollo)
```bash
# Instalar PyTorch primero (importante)
pip install torch torchvision torchaudio

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno (opcional)
export GEMINI_API_KEY="tu-api-key-aqui"
export HF_TOKEN="tu-huggingface-token-aqui"

# Ejecutar la aplicación
streamlit run app.py
```

### 🔧 **Solución de Problemas**

**Si PyTorch no funciona:**
- **Streamlit Cloud**: Contactar soporte de Streamlit
- **HF Spaces**: Verificar logs de construcción
- **Local**: Instalar PyTorch manualmente

**Si faltan dependencias:**
- Ejecutar: `pip install --upgrade pip`
- Reinstalar: `pip install -r requirements.txt --force-reinstall`

**Si hay errores de memoria:**
- Reducir tamaño de imagen de entrada
- Usar CPU en lugar de GPU

##  Guía de Uso

### 🖼️ **Subida de Imagen**
- Formatos soportados: JPG, JPEG, PNG
- Tamaño máximo recomendado: 2MB
- Resolución óptima: Hasta 1024px de lado más largo

### 🎨 **Selección de Modo**

#### **Modo Básico (Recomendado)**
- Configuración automática optimizada
- Pipeline: Real-ESRGAN + CodeFormer + Reducción de ruido suave
- Ideal para usuarios principiantes

#### **Modo Avanzado**
- Control completo con presets inteligentes
- **🔧 Parámetros de Algoritmos**: Sección dedicada para ajustar parámetros de modelos
- Acceso a todos los algoritmos de reparación
- Configuración conservadora por defecto

### ⚙️ **Configuración Avanzada**

#### **🔧 Parámetros de Algoritmos (Nuevo en Modo Avanzado)**
- **🎨 Modelos de Restauración Facial**:
  - **CodeFormer**: Fidelidad 0.1-0.9 (0.7 recomendado), Upscale 1-4x
  - **GFPGAN**: Upscale 1-4x (1x recomendado para evitar distorsiones)
- **🖼️ Mejoras de Imagen**:
  - **Real-ESRGAN**: Modelos x4plus/x4plus-anime
  - **Stable Diffusion**: Fuerza 0.1-1.0 (0.5 para cambios naturales), Pasos 10-50
- **🔧 Procesamiento Avanzado**:
  - **Reducción de ruido**: Intensidad 1-5 (1 para detalles finos)
  - **Eliminación de arañazos**: Sensibilidad 1-10 (2 conservador)
  - **Retoque de color**: Intensidad 0.5-2.0 (0.7 sutil)

### 📊 **Análisis de Resultados**
- **Comparación visual** lado a lado
- **Métricas de cambio** (diferencia pixel promedio)
- **Análisis CLIP** de contenido y calidad
- **Análisis Gemini** comparativo detallado

## 🔧 Arquitectura Técnica

### 📁 Estructura del Proyecto
```
restaurador-mvp/
├── app.py                    # 🖥️ Aplicación principal Streamlit
├── models/
│   ├── __init__.py          # 📦 Inicialización del módulo
│   ├── analysis.py          # 🤖 Funciones de análisis con IA
│   └── diffusion.py         # 🎨 Algoritmos de restauración
├── utils/
│   ├── __init__.py          # 📦 Inicialización del módulo
│   ├── image_utils.py       # 🖼️ Utilidades de procesamiento de imágenes
│   └── ui_utils.py          # 🎛️ Utilidades de interfaz
├── .streamlit/
│   └── config.toml          # ⚙️ Configuración de Streamlit
├── assets/                   # 📸 Imágenes de ejemplo
├── gfpgan/                  # 🎭 Modelos GFPGAN
├── CodeFormer.pth           # 🎨 Modelo CodeFormer
├── RealESRGAN_x4plus.pth    # 🖼️ Modelo Real-ESRGAN
├── Dockerfile               # 🐳 Configuración Docker
├── requirements.txt         # 📦 Dependencias Python
└── README.md               # 📖 Esta documentación
```

### 🏗️ Pipeline de Procesamiento
1. **Upscaling** (Real-ESRGAN) - Base de calidad
2. **Restauración Facial** (CodeFormer/GFPGAN) - Rostros primero
3. **Contraste Local** (CLAHE) - Definición adaptativa
4. **Retoque de Color** - Ajustes tonales
5. **Reducción de Ruido** - Limpieza de granulado
6. **Afilado** - Realce de detalles
7. **Reparación Física** - Arañazos y manchas
8. **Acabados** - Bordes y colorización
9. **IA Creativa** (Stable Diffusion) - Toque final opcional

## 🎯 Limitaciones y Consideraciones

### ⚠️ Limitaciones de la IA
- **Daños físicos profundos** no se pueden eliminar completamente
- **Información perdida** no se puede reconstruir
- **Resultados variables** según calidad de imagen original
- **Tiempo de procesamiento** depende de algoritmos seleccionados

### 💡 Recomendaciones
- **Fotos con daños menores** obtienen mejores resultados
- **Imágenes nítidas** con rostros visibles funcionan mejor
- **Configuración conservadora** produce resultados más naturales
- **Prueba diferentes ajustes** para encontrar el equilibrio óptimo

## 🤝 Contribución

### 🚀 Mejoras Futuras
- [ ] Soporte para video
- [ ] Modelos de colorización avanzados
- [ ] Interfaz móvil optimizada
- [ ] Procesamiento por lotes
- [ ] Modelos personalizados

### 🐛 Reporte de Problemas

#### Para Spaces de Hugging Face:
Si encuentras errores en el despliegue:
1. **Revisa los logs de construcción** en la pestaña "Build" del Space
2. **Verifica que todos los archivos** estén presentes en el repositorio
3. **Comprueba las variables de entorno** (Secrets) si usas APIs
4. **Reporta issues** en el repositorio del proyecto

#### Para instalación local:
Si encuentras errores o tienes sugerencias:
1. Revisa los logs de la aplicación
2. Verifica la configuración de APIs
3. Prueba con diferentes imágenes
4. Reporta en la sección de issues

### 🔧 Solución de Problemas Comunes

#### ❌ "PyTorch no está disponible"
- **En HF Spaces**: El Space puede estar usando una configuración incompatible
- **Solución**: Espera a que HF Spaces complete la instalación o contacta soporte
- **Alternativa**: La app funciona en modo básico sin PyTorch

#### ❌ "Error al importar módulos"
- **Causa**: Dependencias faltantes o archivos corruptos
- **Solución**: Asegúrate de que todos los archivos del proyecto estén presentes

#### ❌ "Modelos no se descargan"
- **En HF Spaces**: Las descargas externas pueden estar bloqueadas
- **Solución**: Los modelos están configurados para usar HF Hub como alternativa

##  Licencia

Este proyecto está bajo la Licencia MIT.

---



