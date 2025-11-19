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

> Transforma fotos antiguas dañadas con IA avanzada y obtén análisis detallado de mejoras.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mvprestauradorfotos.streamlit.app/)

## Descripción

Esta aplicación web utiliza inteligencia artificial de vanguardia para restaurar fotos antiguas dañadas, aplicando múltiples algoritmos de procesamiento de imágenes. Está diseñada para fotógrafos amateurs, historiadores y cualquier persona con colecciones de fotos vintage que deseen recuperar su calidad original.

**Accede a la aplicación en vivo:** [https://mvprestauradorfotos.streamlit.app/](https://mvprestauradorfotos.streamlit.app/)

El sistema combina modelos de super-resolución, restauración facial, reducción de ruido y análisis inteligente para ofrecer resultados profesionales. Utiliza tecnologías como PyTorch para procesamiento eficiente y Streamlit para una interfaz intuitiva.

Lo que diferencia esta herramienta es su enfoque integral: no solo restaura, sino que también analiza los cambios realizados, proporcionando feedback detallado sobre las mejoras aplicadas.

## User Persona

**Nombre**: Susana

**Edad**: 60-70 años

**Ocupación**: Jubilada (anteriormente administrativa)

**Contexto tecnológico**: Nivel usuario básico. Usa fluidamente WhatsApp y Facebook para comunicarse con su familia. Sabe navegar por páginas web, pero se frustra e intimida fácilmente si una interfaz tiene "demasiados botones", opciones o jerga técnica que no entiende. No tiene idea de qué es la IA.

**Problema actual**:
"Estoy digitalizando todas las fotos viejas de la familia. Tengo esta única foto del casamiento de mis abuelos de 1950. Está rayada, muy borrosa y, obviamente, en blanco y negro. Me da mucha pena porque apenas se les ven las caras, y me encantaría poder verlos con más claridad y quizás imaginar cómo se veía en color."

**Solución actual**:
Intentó usar los "filtros automáticos" de la aplicación de fotos de su celular, pero "no hicieron nada". Buscó en Google "restaurar fotos" y encontró servicios profesionales que le cobran demasiado por cada foto, o programas de edición que parecen "hechos para diseñadores".

**Frustraciones**:
- Los programas de edición (como Photoshop) son "imposibles de entender" y le parecen una pérdida de tiempo.
- Le frustra que los filtros de su celular sean "para ponerle orejas de perro a las fotos" en lugar de arreglar problemas reales.

**Objetivos**:
- "Quiero ver las caras de mis abuelos con claridad, como si la foto fuera nueva. Poder verles los ojos, la sonrisa."
- "El éxito para mí es poder descargar la foto arreglada, mandarla al grupo de WhatsApp de la familia y que mis primos me digan '¡Qué increíble! ¿Cómo hiciste?'".

**Contexto de uso**:
Usaría el sistema un fin de semana por la tarde, en su casa, usando la computadora de escritorio (no el celular). Espera una página web simple: subir un archivo, apretar un botón que diga "Restaurar" o "Arreglar", y ver el "Antes" y "Después". Quiere un botón claro de "Descargar" para guardar la foto nueva.

"Mi sistema permite que Susana (una jubilada que digitaliza su archivo fotográfico familiar) pueda restaurar la claridad de rostros borrosos en fotos antiguas aplicando restauración facial con GFPGAN (o CodeFormer) y verificando resultados con Gemini 2.0 para describir cualitativamente las mejoras."

## Demo

![Demo Screenshot](assets/ejemplos/ejemplo_antiguo.jpg)

[Link al video demo (2-3 minutos)](https://example.com/demo-video)  <!-- Reemplazar con link real -->

## Características

- Procesamiento con Real-ESRGAN, CodeFormer, GFPGAN y Stable Diffusion
- Análisis visual con Gemini AI 2.0 y CLIP classification
- Interfaz intuitiva en Streamlit con modos básico y avanzado
- Comparación lado a lado de resultados originales vs restaurados
- 9 algoritmos de restauración configurables
- Detección automática de cambios con métricas de calidad

## Tecnologías Utilizadas

**Frontend:**
- Streamlit 1.39.0

**Modelos de IA:**
- Real-ESRGAN - Para upscaling x4 de ultra-alta calidad
- CodeFormer - Para restauración facial inteligente sin recortes
- GFPGAN - Para restauración facial alternativa con recortes
- Stable Diffusion - Para mejoras creativas con IA generativa
- Gemini AI - Para análisis comparativo detallado
- CLIP - Para análisis automático de contenido y calidad

**Procesamiento:**
- PIL/Pillow para manipulación de imágenes
- NumPy para operaciones matriciales
- OpenCV para procesamiento de visión por computadora
- PyTorch para deep learning

**Deployment:**
- Streamlit Cloud (recomendado)
- Hugging Face Spaces (alternativa)

## Arquitectura del Sistema

```
Usuario → Streamlit UI → Selección de Modo/Algoritmos → Procesamiento Pipeline → Análisis IA → Resultados
```

1. **Interfaz de Usuario**: Streamlit maneja la subida de imágenes y configuración de parámetros
2. **Pipeline de Procesamiento**: Aplica algoritmos en secuencia (upscaling → restauración facial → mejoras → análisis)
3. **Modelos de IA**: Ejecutan el procesamiento pesado usando GPU/CPU
4. **Análisis**: Gemini y CLIP evalúan los resultados y generan explicaciones
5. **Salida**: Comparación visual y métricas de calidad

## Instalación Local

### Requisitos Previos
- Python 3.10+
- GPU NVIDIA recomendada (mejora rendimiento significativamente)
- 4GB+ RAM
- API Keys: GEMINI_API_KEY (Google AI), HF_TOKEN (Hugging Face)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/Nulltheory/Tecnicas_de_Procesamiento_de_Imagenes.git
cd Tecnicas_de_Procesamiento_de_Imagenes/MVP_Restaurador_de_Fotos
```

2. Crear entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
# Crear archivo .env o usar export
GEMINI_API_KEY=tu_clave_de_google_ai_aqui
HF_TOKEN=tu_token_de_huggingface_aqui
```

5. Ejecutar aplicación:
```bash
streamlit run app.py
```

## Uso

1. Abrí la aplicación en http://localhost:8501 o en el enlace de deployment
2. Subí una imagen en formato JPG, JPEG o PNG (máximo 2MB recomendado)
3. Seleccioná el modo de funcionamiento:
   - **Básico**: Configuración automática optimizada
   - **Avanzado**: Control completo de parámetros
4. Ajustá parámetros específicos si usás modo avanzado
5. Hacé click en "Procesar Imagen"
6. Revisá los resultados en la comparación lado a lado
7. Leé el análisis detallado generado por IA
8. Descargá la imagen restaurada si estás satisfecho

![Paso 1: Subida](assets/ejemplos/ejemplo_antiguo.jpg)
![Paso 2: Procesamiento](assets/ejemplos/ejemplo_antiguo.jpg)
![Paso 3: Resultados](assets/ejemplos/ejemplo_antiguo.jpg)

## Decisiones de Diseño

### Por qué Streamlit
Elegí Streamlit por su simplicidad para crear prototipos rápidos y su capacidad de generar interfaces web nativas sin necesidad de conocimientos avanzados de frontend. Facilita el deployment y mantiene la transparencia del código, alineándose con principios de desarrollo ágil y accesibilidad.

### Por qué estos modelos de IA específicos
Seleccioné una combinación de modelos open-source probados (Real-ESRGAN, CodeFormer) por su balance óptimo entre calidad de resultados, velocidad de procesamiento y accesibilidad. Todos están disponibles en Hugging Face Hub, evitando descargas externas problemáticas.

### Por qué análisis con IA generativa
El análisis con Gemini AI proporciona explicabilidad y valor añadido, transformando una simple herramienta técnica en una experiencia educativa. Permite al usuario entender qué cambios se realizaron y por qué, siguiendo principios de Human-AI Interaction.

### Principios de Human-AI Interaction Aplicados

**Transparencia**: La aplicación muestra claramente qué algoritmos se aplican, con qué parámetros y en qué orden. Los usuarios pueden ver el "pipeline" completo.

**Control**: Tres niveles de control (básico, avanzado, manual) permiten que usuarios de diferentes niveles técnicos puedan usar la herramienta efectivamente.

**Explicabilidad**: El análisis con Gemini proporciona explicaciones detalladas en lenguaje natural sobre qué mejoras se realizaron y por qué.

**Manejo de errores**: Mensajes claros cuando hay problemas (ej: "Imagen demasiado grande"), con sugerencias específicas de solución.

Referencias:
- IBM AI Product Design: https://www.ibm.com/think/topics/ai-product-design
- IxDF Human-AI Interaction: https://www.interaction-design.org/literature/topics/human-ai-interaction

## Conceptos de Procesamiento Digital Aplicados

- **Filtrado espacial**: Reducción de ruido gaussiano y eliminación de arañazos mediante filtros de mediana y detección de bordes
- **Transformaciones de intensidad**: Ecualización de histograma adaptativa (CLAHE) para mejorar contraste local, y unsharp masking para realce de detalles
- **Restauración de imágenes**: Super-resolution con redes convolucionales (Real-ESRGAN), inpainting con modelos de difusión (Stable Diffusion)
- **Análisis de imágenes**: Clasificación semántica con CLIP, métricas de calidad PSNR/SSIM para evaluación objetiva

## Limitaciones Conocidas

- Funciona mejor con fotos que contienen rostros humanos y escenas bien iluminadas
- Tiempo de procesamiento típico: 30-120 segundos dependiendo de algoritmos seleccionados y hardware
- Tamaño máximo de imagen recomendado: 2MB, resolución hasta 1024px de lado más largo
- No puede restaurar información completamente perdida (ej: texto borrado, objetos faltantes)
- Requiere conexión a internet estable para descargar modelos desde Hugging Face Hub
- Resultados variables según la calidad y tipo de daño de la imagen original

## Métricas de Calidad

- **PSNR (Peak Signal-to-Noise Ratio)**: Mide la diferencia promedio entre píxeles originales y restaurados. Valores superiores a 30 dB indican buena calidad de restauración.
- **SSIM (Structural Similarity Index)**: Evalúa la similitud estructural manteniendo detalles importantes. Valores superiores a 0.8 son considerados aceptables para restauración.

**Interpretación para usuarios**: Un aumento en PSNR indica mejora técnica, mientras que mantener SSIM alto asegura que la imagen restaurada preserve la estructura original sin distorsiones artificiales.

## Trabajo Futuro

- [ ] Soporte para restauración de videos cortos
- [ ] Integración de modelos avanzados de colorización automática
- [ ] Optimización de interfaz para dispositivos móviles
- [ ] Funcionalidad de procesamiento por lotes
- [ ] Entrenamiento de modelos personalizados con datos del usuario

## Reflexiones y Aprendizajes

Construir esta aplicación me permitió profundizar en la integración práctica de múltiples modelos de IA en un sistema coherente. Los desafíos técnicos principales fueron la gestión eficiente de memoria con PyTorch en entornos con recursos limitados y la optimización de pipelines de procesamiento para mantener tiempos de respuesta aceptables.

Utilicé herramientas de IA generativa (ChatGPT, Claude) para acelerar el desarrollo inicial del código base, generación de documentación y resolución de problemas específicos. Esto me permitió enfocarme en la lógica de negocio y el diseño de experiencia de usuario.

La próxima vez planificaría mejor la arquitectura modular desde el inicio, implementando una mejor separación de responsabilidades entre componentes. También invertiría más tiempo en el diseño de la experiencia de usuario, incluyendo pruebas con usuarios reales para validar las decisiones de diseño.

## Recursos y Referencias

- [Documentación oficial de Streamlit](https://docs.streamlit.io/)
- [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- [CodeFormer: Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://arxiv.org/abs/2206.11253)
- [Hugging Face Model Hub](https://huggingface.co/models)
- Notebooks del curso IFTS 24 - Procesamiento Digital de Imágenes

## Autor

**Alejandro**  
Estudiante de Tecnicatura Superior en Ciencias de Datos e IA - IFTS 24  
Materia: Procesamiento Digital de Imágenes e Introducción a Visión por Computadora  
Año: 2025  

[GitHub](https://github.com/Nulltheory) | [LinkedIn](https://linkedin.com/in/alejandro-profile)

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Agradecimientos

- **Profesor**: Matías Barreto por la guía y conocimientos transmitidos en la materia
- **Comunidad Open Source**: Por los modelos de IA disponibles que hicieron posible este proyecto
- **Herramientas de IA**: ChatGPT, Claude y GitHub Copilot por asistencia en desarrollo y documentación
- **Hugging Face**: Por la plataforma que facilita el acceso a modelos de IA

---

**Trabajo Integrador N°2 - IFTS 24 - 2025**




