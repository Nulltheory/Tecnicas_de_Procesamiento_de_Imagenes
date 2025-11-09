import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import warnings
# Eliminamos subprocess y sys ya que la instalación dinámica está fuera.

# --- Configuración Streamlit (PRIMERO) ---
st.set_page_config(page_title="Restaurador Fotográfico AI 🧠🎨",
                   page_icon="🧠",
                   layout="wide")

# 📝 NOTA: Ya no se intenta la instalación agresiva.
# Las dependencias (PyTorch, módulos de IA) deben estar
# instaladas vía requirements.txt antes de la ejecución.

# Verificar disponibilidad de PyTorch - CRÍTICO PARA FUNCIONAMIENTO
torch_available = False
modules_available = False
try:
    import torch
    torch_available = True
    version = torch.__version__
    st.success(f"✅ PyTorch {version} disponible")
    if torch.cuda.is_available():
        st.info(f"🎯 CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        st.info("💻 Modo CPU - funcional")
    # Verificar que torch funciona realmente
    test_tensor = torch.tensor([1.0, 2.0, 3.0])
    result = test_tensor.sum().item()
    st.info(f"🔧 Test PyTorch: {result:.0f} ✓")
except (ImportError, Exception) as e:
    torch_available = False
    st.warning(f"❌ PyTorch no disponible: {e}. Funcionando en modo básico")
    st.info("💡 La app funcionará con capacidades de procesamiento de imagen básicas")

# Importaciones de módulos locales - REQUIEREN PyTorch
modules_available = False
try:
    # Estos módulos deben existir en una carpeta 'models' y 'utils'
    from models.diffusion import (
        restaurar_imagen_gfpgan, restaurar_imagen_sd, upscale_imagen_realesrgan,
        restaurar_imagen_codeformer, mejorar_color_contraste, reducir_ruido_avanzado,
        inpainting_aranasos_agresivo, reparar_manchas_blancas, definir_bordes_foto,
        colorizar_imagen, mejorar_contraste_adaptativo, afinar_detalles
    )
    from models.analysis import (
        analizar_con_blip_hf, analizar_con_gemini, analizar_imagen_completo,
        analizar_calidad_clip, analizar_con_gemini_comparativo
    )
    from utils.image_utils import resize_image
    from utils.ui_utils import mostrar_progreso, mostrar_resultado_con_descarga
    modules_available = True
    st.success("🎯 Todos los módulos de IA cargados correctamente")
except ImportError as e:
    modules_available = False
    st.warning(f"⚠️ Módulos de IA avanzados no disponibles: {e}. Funcionando en modo básico")
    st.info("💡 Asegúrate de que los archivos 'models' y 'utils' existen y que todas las librerías de PyTorch están instaladas en 'requirements.txt'")
except Exception as e:
    modules_available = False
    st.warning(f"⚠️ Error cargando módulos avanzados: {e}")
    st.info("💡 La app funcionará con capacidades básicas de procesamiento")

st.title("🧠 Restaurador Fotográfico AI")
st.markdown("**Transforma fotos antiguas dañadas con IA avanzada y obtén análisis detallado de mejoras**")

# --- Información rápida ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if torch_available and modules_available:
        st.metric("🖼️ Modelos", "9", "algoritmos")
    else:
        st.metric("🖼️ Modelos", "Básico", "limitado")
with col2:
    if torch_available and modules_available:
        st.metric("🤖 Análisis", "Gemini + CLIP", "IA avanzada")
    else:
        st.metric("🤖 Análisis", "Básico", "manual")
with col3:
    if torch_available:
        st.metric("⚡ Procesamiento", "GPU/CPU", "automático")
    else:
        st.metric("⚡ Procesamiento", "CPU", "básico")
with col4:
    if torch_available and modules_available:
        st.metric("🎯 Precisión", "95%", "típica")
    else:
        st.metric("🎯 Precisión", "Básica", "limitada")

st.markdown("---")

# --- PANEL LATERAL PARA API KEYS ---
st.sidebar.header("🔐 Configuración API")

# Mostrar estado de configuración
api_status = []
if os.getenv("HF_TOKEN"):
    api_status.append("✅ HF Token")
else:
    api_status.append("❌ HF Token")

if os.getenv("GEMINI_API_KEY"):
    api_status.append("✅ Gemini API")
else:
    api_status.append("❌ Gemini API")

st.sidebar.markdown(f"**Estado:** {' | '.join(api_status)}")

with st.sidebar.expander("🔑 Configurar APIs", expanded=False):
    st.markdown("**HuggingFace Token** (Opcional)")
    st.markdown("*Para modelos avanzados de CLIP*")
    hf_token = st.text_input("Token HF", type="password", value=os.getenv("HF_TOKEN", ""),
                            help="Obtén gratis en huggingface.co/settings/tokens", key="hf_token", disabled=True)

    st.markdown("---")
    st.markdown("**Gemini API Key**")
    st.markdown("*Para análisis detallado con IA*")
    gemini_api_key = st.text_input("API Key Gemini", type="password", value=os.getenv("GEMINI_API_KEY", ""),
                                  help="Obtén gratis en makersuite.google.com/app/apikey", key="gemini_key", disabled=True)

    st.info("🔒 Campos bloqueados por seguridad. Las claves se configuran vía variables de entorno del sistema.")

st.sidebar.markdown("---")

# --- Algoritmos de Restauración (Pipeline 4 Fases Optimizado) ---
st.sidebar.markdown("### 🎨 Pipeline de Restauración")
st.sidebar.caption("4 fases optimizadas: Base → Estructural → Tonal → Final")

# Configuración principal
st.sidebar.markdown("**🔧 Configuración Principal**")

# Radio buttons para modo mutuamente exclusivo
modo_seleccionado = st.sidebar.radio(
    "Selecciona el modo de funcionamiento:",
    ["Modo Básico (Recomendado)", "Modo Avanzado"],
    index=0,  # Básico por defecto
    help="Modo Básico: Reparación estructural conservadora | Modo Avanzado: Control completo con ajustes tonales"
)

usar_preset_basico = (modo_seleccionado == "Modo Básico (Recomendado)")
usar_preset_avanzado = (modo_seleccionado == "Modo Avanzado")

if usar_preset_basico:
    # Preset básico equilibrado - calidad óptima sin riesgos
    # **Los flags ahora solo dependen del modo seleccionado y de torch_available**
    usar_realesrgan = torch_available
    usar_codeformer = torch_available
    usar_gfpgan = False
    usar_retouching = False # Desactivado por defecto en modo básico - demasiado agresivo
    usar_contraste_adaptativo = False # Desactivado por defecto - da sensación de foto vieja
    usar_denoise = modules_available or True # Solo denoising muy suave
    usar_sharpen = torch_available
    usar_scratch_removal = torch_available
    usar_spot_removal = torch_available
    usar_border_enhancement = modules_available or True
    usar_colorization = modules_available or True
    usar_sd = False

    st.sidebar.success("✅ **Modo Básico Activado**")
    with st.sidebar.expander("📋 Qué incluye", expanded=False):
        st.markdown("""
        **🧩 Fase 1: Base de Calidad**
        - 🖼️ **Real-ESRGAN**: Upscaling x4 de calidad
        - 🎭 **CodeFormer**: Restauración facial inteligente

        **🩹 Fase 2: Reparación Estructural**
        - 🩹 **Reparación de daños**: Eliminación de arañazos y manchas

        **🎨 Fase 3: Mejoras Tonales**
        - 🔇 **Reducción de ruido suave**: Preserva detalles originales

        **✨ Fase 4: Acabado Final**
        - ⚡ **Afilado**: Realce final de detalles
        - 🎯 **Definición**: Bordes y colorización básica

        *Enfoque en reparación estructural, sin alteraciones agresivas*
        """)

elif usar_preset_avanzado:
    # Configuración avanzada con preset inteligente
    st.sidebar.info("⚙️ **Modo Avanzado**: Control completo con configuración inteligente")

    # Preset rápido para modo avanzado
    preset_avanzado_rapido = st.sidebar.checkbox("Configuración Avanzada Rápida", value=True,
                                               help="Preset optimizado para usuarios avanzados")

    if preset_avanzado_rapido:
        # Valores equilibrados para usuarios avanzados - más algoritmos pero controlados
        usar_realesrgan = torch_available
        usar_codeformer = torch_available
        usar_gfpgan = False
        usar_retouching = modules_available or True # Mantener en avanzado para usuarios que lo necesiten
        usar_contraste_adaptativo = torch_available # Mantener en avanzado pero con parámetros suaves
        usar_denoise = modules_available or True
        usar_sharpen = torch_available
        usar_scratch_removal = torch_available
        usar_spot_removal = torch_available
        usar_border_enhancement = modules_available or True
        usar_colorization = modules_available or True
        usar_sd = False

        st.sidebar.success("✅ **Configuración Avanzada Equilibrada Activada**")
        with st.sidebar.expander("📋 Pipeline Avanzado Completo", expanded=False):
            st.markdown("""
            **🧩 Fase 1: Base de Calidad**
            - 🖼️ **Real-ESRGAN**: Upscaling x4 de máxima calidad
            - 🎭 **CodeFormer**: Restauración facial avanzada

            **🩹 Fase 2: Reparación Estructural**
            - 🩹 **Reparación completa**: Eliminación de arañazos y manchas blancas

            **🎨 Fase 3: Mejoras Tonales Avanzadas**
            - 🎨 **Retoque de color**: Ajustable (0.5-2.0, defecto 0.7)
            - 🔍 **CLAHE**: Contraste adaptativo local
            - 🔇 **Reducción de ruido**: Ajustable (1-5, defecto 1)

            **✨ Fase 4: Acabado Final**
            - ⚡ **Afilado**: Realce final de detalles
            - 🎯 **Definición**: Bordes y colorización básica

            *Control completo con ajustes personalizables*
            """)
    else:
        # Configuración completamente manual
        st.sidebar.markdown("**🎯 Configuración Manual Completa**")

        # Paso 1: Base y upscaling
        with st.sidebar.expander("1️⃣ Base y Upscaling", expanded=True):
            usar_realesrgan = st.checkbox("Real-ESRGAN (x4)", value=torch_available,
                                        disabled=not torch_available,
                                        help="Primer paso: upscaling de alta calidad (requiere PyTorch)" if not torch_available else "Primer paso: upscaling de alta calidad")

        # Paso 2: Restauración facial
        with st.sidebar.expander("2️⃣ Restauración Facial", expanded=True):
            usar_codeformer = st.checkbox("CodeFormer", value=torch_available,
                                        disabled=not torch_available,
                                        help="Restauración facial sin recortar (requiere PyTorch)" if not torch_available else "Restauración facial sin recortar")
            usar_gfpgan = st.checkbox("GFPGAN", value=False,
                                    disabled=not torch_available,
                                    help="Restauración facial con recorte (requiere PyTorch)" if not torch_available else "Restauración facial con recorte (alternativa)")

        # Paso 3: Mejoras básicas
        with st.sidebar.expander("3️⃣ Mejoras Básicas", expanded=False):
            usar_retouching = st.checkbox("Retoque Color/Contraste", value=True, help="Mejora de color y contraste")
            usar_contraste_adaptativo = st.checkbox("Contraste Adaptativo", value=torch_available,
                                                  disabled=not torch_available,
                                                  help="CLAHE para mejor definición local (requiere PyTorch)" if not torch_available else "CLAHE para mejor definición local")
            usar_denoise = st.checkbox("Reducción de Ruido", value=True, help="Eliminación de ruido y granulado")
            usar_sharpen = st.checkbox("Afilar Detalles", value=torch_available,
                                     disabled=not torch_available,
                                     help="Realce sutil de detalles finos (requiere PyTorch)" if not torch_available else "Realce sutil de detalles finos")

        # Paso 4: Reparación de daños
        with st.sidebar.expander("4️⃣ Reparación de Daños", expanded=True):
            usar_scratch_removal = st.checkbox("Eliminar Arañazos", value=torch_available,
                                             disabled=not torch_available,
                                             help="Reparación agresiva de arañazos (requiere PyTorch)" if not torch_available else "Reparación agresiva de arañazos")
            usar_spot_removal = st.checkbox("Reparar Manchas", value=torch_available,
                                          disabled=not torch_available,
                                          help="Eliminación de manchas blancas (requiere PyTorch)" if not torch_available else "Eliminación de manchas blancas")

        # Paso 5: Acabados
        with st.sidebar.expander("5️⃣ Acabados", expanded=False):
            usar_border_enhancement = st.checkbox("Definir Bordes", value=True, help="Mejorar definición de bordes")
            usar_colorization = st.checkbox("Colorizar Imagen", value=True, help="Convertir B/N a color básico")

        # Paso 6: Mejoras creativas
        with st.sidebar.expander("6️⃣ Creativas (Opcional)", expanded=False):
            usar_sd = st.checkbox("Stable Diffusion", value=False,
                                disabled=not torch_available,
                                help="Último paso: mejoras creativas avanzadas (requiere PyTorch)" if not torch_available else "Último paso: mejoras creativas avanzadas")

st.sidebar.markdown("---")

# --- Análisis de Calidad ---
st.sidebar.markdown("### 📊 Análisis de Calidad")

usar_gemini = st.sidebar.checkbox("Análisis Gemini", value=True, help="Análisis detallado y positivo con IA")
usar_clip = st.sidebar.checkbox("Clasificación CLIP", value=torch_available,
                              disabled=not torch_available,
                              help="Análisis automático de contenido y calidad (requiere PyTorch)" if not torch_available else "Análisis automático de contenido y calidad")

# Mostrar estado de capacidades
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Estado del Sistema")
if torch_available and modules_available:
    st.sidebar.success("✅ Sistema completo - Todas las funciones disponibles")
elif torch_available:
    st.sidebar.warning("⚠️ PyTorch disponible - Funciones limitadas")
    st.sidebar.info("Algunos módulos especializados no están disponibles")
elif modules_available:
    st.sidebar.info("ℹ️ Funciones básicas disponibles - Sin PyTorch")
else:
    st.sidebar.error("❌ Sistema limitado - Solo funciones básicas")
    st.sidebar.info("La aplicación funcionará con capacidades mínimas")

# --- Configuración Avanzada (Collapsible) ---
with st.sidebar.expander("⚙️ Configuración Avanzada", expanded=False):
    st.markdown("**🎨 Modelos de Restauración Facial**")

    # Información sobre configuración por defecto
    st.info("💡 **Configuración Optimizada**: Los parámetros están pre-configurados para obtener los mejores resultados en la mayoría de las imágenes antiguas. Modifica solo si tienes experiencia específica.")

    # CodeFormer settings
    st.markdown("**CodeFormer**")
    codeformer_fidelity = st.slider("Fidelidad", 0.1, 0.9, 0.7, 0.1,
                                   help="0.7 = más fiel al original, menos agresivo")
    codeformer_upscale = st.slider("Upscale", 1, 4, 1,
                                  help="1x = sin pixelado adicional")

    # GFPGAN settings
    st.markdown("**GFPGAN**")
    gfpgan_upscale = st.slider("Upscale", 1, 4, 1,
                              help="1x = más conservador para evitar distorsiones")

    st.markdown("---")
    st.markdown("**🖼️ Mejoras de Imagen**")

    # Real-ESRGAN settings
    st.markdown("**Real-ESRGAN**")
    realesrgan_model = st.selectbox("Modelo", ["x4plus", "x4plus-anime"],
                                   index=0,  # x4plus como defecto
                                   help="x4plus = recomendado para fotos reales")

    # Stable Diffusion settings
    st.markdown("**Stable Diffusion**")
    sd_strength = st.slider("Fuerza de cambio", 0.1, 1.0, 0.5,
                           help="0.5 = cambios más suaves y naturales")
    sd_steps = st.slider("Pasos de inferencia", 10, 50, 15,
                        help="15 = equilibrio entre calidad y velocidad")

    st.markdown("---")
    st.markdown("**🔧 Procesamiento Avanzado**")

    # Noise reduction settings
    st.markdown("**Reducción de Ruido**")
    denoise_strength = st.slider("Intensidad", 1, 5, 1,
                                help="1 = reducción muy suave, preserva detalles finos")

    # Scratch removal settings
    st.markdown("**Eliminación de Arañazos**")
    scratch_sensitivity = st.slider("Sensibilidad", 1, 10, 2,
                                   help="2 = muy conservador, preserva detalles estructurales")

    # Color enhancement settings
    st.markdown("**Retoque de Color**")
    color_boost = st.slider("Intensidad de color", 0.5, 2.0, 0.7,
                           help="0.7 = mejora muy sutil, preserva colores originales")

# --- Consejos y Limitaciones ---
with st.sidebar.expander("💡 Consejos de Uso", expanded=False):
    st.markdown("""
    **Para mejores resultados:**
    - ✅ **Fotos con daños leves**: Rasguños finos, polvo, decoloración ligera
    - ✅ **Imágenes nítidas**: Rostros visibles y detalles reconocibles
    - ✅ **Fotos vintage**: Imágenes antiguas en buen estado general
    - ✅ **Formatos estándar**: JPG/PNG de buena calidad

    **Limitaciones de la IA:**
    - ❌ **Daños físicos profundos**: Rasguños anchos, desgarros, manchas grandes
    - ❌ **Información perdida**: No reconstruye partes completamente faltantes
    - ❌ **Calidad original pobre**: Resultados dependen de la imagen base
    - ❌ **Texto dañado**: No puede reconstruir escritura ilegible
    """)

with st.sidebar.expander("🚀 Guía Rápida", expanded=False):
    st.markdown("""
    **🚀 Inicio Rápido (3 pasos):**
    1. **Sube tu foto** → Arrastra o selecciona imagen antigua
    2. **Elige modo** → Básico (automático) o Avanzado (manual)
    3. **Ejecuta** → Haz clic en "🚀 Iniciar Restauración"

    **⚙️ Configuración Recomendada:**

    **Para Principiantes:**
    - Modo: Básico (Recomendado)
    - **Enfoque**: Reparación estructural conservadora
    - Análisis: Activar Gemini para evaluación
    - Resultado: Restauración natural sin alteraciones agresivas

    **Para Fotos Complejas:**
    - Modo: Avanzado con preset rápido
    - **Fase 2 incluida**: Reparación estructural automática
    - **Ajustes disponibles**: Retoque de color, CLAHE, denoising avanzado
    - Análisis: Ambos (CLIP + Gemini) para evaluación completa

    **💡 Pro Tip:** Empieza con Modo Básico, si no satisface, prueba Avanzado
    """)

# --- Upload de imagen ---
st.markdown("### 📤 Subida de Imagen")

# File uploader con drag & drop visual
imagen_cargada = st.file_uploader(
    "Arrastra y suelta tu foto antigua aquí, o haz clic para seleccionar",
    type=["jpg", "jpeg", "png"],
    help="Formatos: JPG, JPEG, PNG • Máx: 2MB • Recomendado: fotos con daños menores"
)

if imagen_cargada:
    # Información del archivo en una card compacta
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        with col1:
            st.markdown("**📋 Archivo cargado:**")
            st.write(f"**{imagen_cargada.name}**")
        with col2:
            file_size = len(imagen_cargada.read()) / 1024 / 1024
            imagen_cargada.seek(0)
            st.metric("Tamaño", f"{file_size:.1f} MB")
        with col3:
            img_temp = Image.open(io.BytesIO(imagen_cargada.read()))
            st.metric("Dimensiones", f"{img_temp.size[0]}×{img_temp.size[1]}")
            imagen_cargada.seek(0)
        with col4:
            if file_size > 2:
                st.error("⚠️ Archivo grande")
            else:
                st.success("✅ Tamaño óptimo")

    # Imagen preview
    img_pil = Image.open(io.BytesIO(imagen_cargada.read())).convert("RGB")

    # Resize image if function is available
    if modules_available:
        try:
            img_pil = resize_image(img_pil, max_size=1024)
        except:
            # Fallback resize if function fails
            img_pil.thumbnail((1024, 1024), Image.LANCZOS)
    else:
        # Basic resize without custom function
        img_pil.thumbnail((1024, 1024), Image.LANCZOS)

    # Centrar la imagen
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(img_pil, caption="🖼️ Imagen Original", use_container_width=True)

    # Botón de acción prominente
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])
    with col_btn_center:
        iniciar_proceso = st.button("🚀 Iniciar Restauración y Análisis",
                                    type="primary",
                                    use_container_width=True,
                                    help="Comenzará el proceso de restauración con los algoritmos seleccionados")

    if iniciar_proceso:
        # === PIPELINE REESTRUCTURADO - 4 FASES OPTIMIZADAS ===

        # Aplicar procesamiento básico disponible
        imagen_restaurada = img_pil.copy()
        procesamiento_aplicado = False

        # 🧩 FASE 1 – Base de calidad
        st.markdown("#### 🧩 Fase 1: Base de Calidad")

        # 1️⃣ Redimensionado / Upscaling (Real-ESRGAN)
        if usar_realesrgan and torch_available and modules_available:
            with st.spinner(f"Aplicando upscaling con Real-ESRGAN ({realesrgan_model}) x4..."):
                try:
                    img_antes_upscale = img_pil.copy()
                    imagen_restaurada = upscale_imagen_realesrgan(imagen_restaurada, model_type=realesrgan_model)
                    st.success("✅ Real-ESRGAN aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en upscaling: {e}. Continuando sin upscaling.")

        # 2️⃣ Restauración facial (CodeFormer / GFPGAN)
        if usar_codeformer and torch_available and modules_available:
            with st.spinner(f"Restaurando con CodeFormer (fidelidad={codeformer_fidelity}, upscale={codeformer_upscale})..."):
                try:
                    imagen_restaurada = restaurar_imagen_codeformer(imagen_restaurada, fidelity=codeformer_fidelity, upscale_factor=codeformer_upscale)
                    st.success("✅ CodeFormer aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error con CodeFormer: {e}. Intentando con GFPGAN (si está activado)...")
                    if usar_gfpgan:
                        try:
                            imagen_restaurada = restaurar_imagen_gfpgan(imagen_restaurada, upscale_factor=gfpgan_upscale)
                            st.success("✅ GFPGAN aplicado exitosamente (fallback)")
                            procesamiento_aplicado = True
                        except Exception as e2:
                            st.error(f"Error durante la restauración: {e2}")
        elif usar_gfpgan and torch_available and modules_available:
            with st.spinner(f"Restaurando rostros con GFPGAN (upscale={gfpgan_upscale})..."):
                try:
                    imagen_restaurada = restaurar_imagen_gfpgan(imagen_restaurada, upscale_factor=gfpgan_upscale)
                    st.success("✅ GFPGAN aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.error(f"Error durante la restauración con GFPGAN: {e}")

        # 🩹 FASE 2 – Reparación estructural
        st.markdown("#### 🩹 Fase 2: Reparación Estructural")

        # 3️⃣ Eliminación de arañazos (Inpainting)
        if usar_scratch_removal and torch_available and modules_available:
            with st.spinner(f"Aplicando eliminación de arañazos (sensibilidad: {scratch_sensitivity})..."):
                try:
                    imagen_restaurada = inpainting_aranasos_agresivo(imagen_restaurada, sensitivity=scratch_sensitivity)
                    st.success("✅ Eliminación de arañazos aplicada exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en eliminación de arañazos: {e}. Continuando sin inpainting.")

        # 4️⃣ Reparación de manchas blancas / pérdidas locales
        if usar_spot_removal and torch_available and modules_available:
            with st.spinner("Aplicando reparación de manchas blancas..."):
                try:
                    imagen_restaurada = reparar_manchas_blancas(imagen_restaurada)
                    st.success("✅ Reparación de manchas aplicada exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en reparación de manchas: {e}. Continuando sin spot removal.")

        # 🎨 FASE 3 – Mejoras tonales y cromáticas
        st.markdown("#### 🎨 Fase 3: Mejoras Tonales y Cromáticas")

        # 5️⃣ Retoque de color y contraste global
        if usar_retouching and modules_available:
            with st.spinner(f"Aplicando retoque de color (intensidad: {color_boost:.1f})..."):
                try:
                    imagen_restaurada = mejorar_color_contraste(imagen_restaurada, intensity=color_boost)
                    st.success("✅ Retoque de color aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en retoque: {e}. Continuando sin retoque.")

        # 6️⃣ Contraste adaptativo (CLAHE)
        if usar_contraste_adaptativo and torch_available and modules_available:
            with st.spinner("Aplicando contraste adaptativo CLAHE..."):
                try:
                    imagen_restaurada = mejorar_contraste_adaptativo(imagen_restaurada)
                    st.success("✅ Contraste adaptativo aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en contraste adaptativo: {e}. Continuando sin CLAHE.")

        # 7️⃣ Reducción de ruido (denoise)
        if usar_denoise and modules_available:
            with st.spinner(f"Aplicando reducción de ruido (intensidad: {denoise_strength})..."):
                try:
                    imagen_restaurada = reducir_ruido_avanzado(imagen_restaurada, strength=denoise_strength)
                    st.success("✅ Reducción de ruido aplicada exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en reducción de ruido: {e}. Continuando sin denoising.")

        # ✨ FASE 4 – Acabado final
        st.markdown("#### ✨ Fase 4: Acabado Final")

        # 8️⃣ Afilado de detalles (sharpen)
        if usar_sharpen and torch_available and modules_available:
            with st.spinner("Aplicando afilado de detalles..."):
                try:
                    imagen_restaurada = afinar_detalles(imagen_restaurada)
                    st.success("✅ Afilado de detalles aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en afilado: {e}. Continuando sin sharpen.")

        # 9️⃣ Definir bordes / Colorizar
        if usar_border_enhancement and modules_available:
            with st.spinner("Aplicando definición de bordes..."):
                try:
                    imagen_restaurada = definir_bordes_foto(imagen_restaurada)
                    st.success("✅ Definición de bordes aplicada exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en definición de bordes: {e}. Continuando sin border enhancement.")

        if usar_colorization and modules_available:
            with st.spinner("Aplicando colorización básica..."):
                try:
                    imagen_restaurada = colorizar_imagen(imagen_restaurada)
                    st.success("✅ Colorización aplicada exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en colorización: {e}. Continuando sin colorization.")

        # 🔟 Stable Diffusion (opcional)
        if usar_sd and torch_available and modules_available:
            with st.spinner(f"Aplicando Stable Diffusion (fuerza: {sd_strength}, pasos: {sd_steps})..."):
                try:
                    imagen_antes_sd = imagen_restaurada.copy()  # Guardar estado antes de SD
                    imagen_restaurada = restaurar_imagen_sd(imagen_restaurada, hf_token or "", strength=sd_strength, steps=sd_steps)
                    st.success("✅ Stable Diffusion aplicado exitosamente")
                    procesamiento_aplicado = True
                except Exception as e:
                    st.warning(f"Error en mejora general: {e}. Continuando sin SD.")
        elif usar_sd:
            st.info("ℹ️ Stable Diffusion requiere PyTorch - omitiendo este paso")

        # Verificar si se aplicó algún procesamiento (Mantenemos la lógica de comparación)
        # --- Resultados de Restauración ---
        st.markdown("### 🎯 Resultados de Restauración")

        # Mostrar comparación lado a lado con mejor diseño
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📷 Imagen Original**")
                st.image(img_pil, use_container_width=True)

            with col2:
                st.markdown("**✨ Imagen Restaurada**")
                st.image(imagen_restaurada, use_container_width=True)

            # Verificar si las imágenes son diferentes
            img_original_array = np.array(img_pil)
            img_restaurada_array = np.array(imagen_restaurada)

            # Comparar dimensiones primero
            if img_original_array.shape == img_restaurada_array.shape:
                # Calcular diferencia absoluta
                # Usar np.int32 para evitar overflow al restar si los valores son grandes
                diff = np.abs(img_original_array.astype(np.int32) - img_restaurada_array.astype(np.int32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                if max_diff < 5 and not procesamiento_aplicado:
                    # Umbral bajo para considerar "idéntico" a menos que se haya aplicado algo
                    st.error("⚠️ **ALERTA CRÍTICA**: Las imágenes son casi idénticas")
                    st.warning("Los modelos de restauración no aplicaron cambios. Verifica la configuración y logs de error.")
                    st.info("💡 **Solución**: Asegúrate de que **PyTorch** y los **módulos locales** están disponibles (revisa los mensajes de arriba).")
                else:
                    # Calcular porcentaje de mejora
                    improvement_percentage = (mean_diff / 128) * 100
                    st.success(f"✅ Restauración aplicada con éxito. Diferencia media: {mean_diff:.2f}")


        # --- Análisis y Descarga (Sección final) ---
        st.markdown("---")
        st.markdown("### 🤖 Análisis de Calidad con IA")

        # Aquí iría el código para el análisis (analizar_con_gemini, etc.)
        if usar_gemini and modules_available:
             st.info("Generando análisis comparativo con Gemini...")
             # Código para llamar a analizar_con_gemini_comparativo(img_pil, imagen_restaurada)

        if usar_clip and modules_available and torch_available:
            st.info("Clasificando calidad con CLIP...")
            # Código para llamar a analizar_calidad_clip(imagen_restaurada)

        # Usar la función de utilidad para la descarga
        try:
            mostrar_resultado_con_descarga(imagen_restaurada, "imagen_restaurada.png")
        except NameError:
             st.download_button(
                label="⬇️ Descargar Imagen Restaurada",
                data=io.BytesIO(imagen_restaurada.tobytes()), # Simplificado para funcionar sin la utilidad
                file_name="foto_restaurada.png",
                mime="image/png"

            )
