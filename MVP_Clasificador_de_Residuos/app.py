"""
Aplicación de Clasificación de Imágenes
Desarrollada para Procesamiento Digital de Imágenes y Visión por Computadora
Autor: Alejandro Schiariti
Año: 2025
"""

import gradio as gr
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ============================================
# CARGAR MODELO PERSONALIZADO
# ============================================

# Solución para compatibilidad de Teachable Machine
def DepthwiseConv2D_personalizada(
    kernel_size, strides=(1, 1), padding="valid", depth_multiplier=1,
    data_format=None, dilation_rate=(1, 1), activation=None,
    use_bias=True, depthwise_initializer="glorot_uniform",
    bias_initializer="zeros", depthwise_regularizer=None,
    bias_regularizer=None, activity_regularizer=None,
    depthwise_constraint=None, bias_constraint=None, **kwargs
):
    kwargs.pop("groups", None)
    return tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, padding=padding,
        depth_multiplier=depth_multiplier, data_format=data_format,
        dilation_rate=dilation_rate, activation=activation,
        use_bias=use_bias, depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint, **kwargs
    )

print("Cargando modelo personalizado...")
modelo_custom = load_model(
    "models/keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": DepthwiseConv2D_personalizada}
)

# ============================================
# CARGAR MODELO CLIP (preentrenado)
# ============================================

# Cargar etiquetas
with open("models/labels.txt", "r") as f:
    etiquetas_custom = [line.strip() for line in f.readlines()]

print(f"Modelo cargado con {len(etiquetas_custom)} clases")

# También cargar modelo preentrenado para comparación
modelo_clip = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)

CATEGORIAS_CLIP = [
    "Plástico", 
    "Vidrio", 
    "Papel", 
    "Cartón", 
    "Orgánico", 
    "Metal"
]

# ============================================
# FUNCIONES DE PROCESAMIENTO
# ============================================

def preprocesar_para_teachable(imagen):
    """
    Preprocesa imagen para Teachable Machine.
    """
    # Redimensionar a 224x224
    imagen_redim = imagen.resize((224, 224))
    
    # Convertir a array
    array_imagen = np.asarray(imagen_redim, dtype=np.float32)
    
    # Añadir dimensión de batch
    array_imagen = array_imagen.reshape(1, 224, 224, 3)
    
    # Normalizar [-1, 1]
    array_imagen = (array_imagen / 127.5) - 1
    
    return array_imagen

def clasificar_con_custom(imagen):
    """
    Clasifica con modelo personalizado.
    """
    if imagen is None:
        return {"Error": 1.0}
    
    try:
        # Preprocesar
        img_procesada = preprocesar_para_teachable(imagen)
        
        # Predecir
        prediccion = modelo_custom.predict(img_procesada, verbose=0)
        
        # Formatear resultados
        resultados = {
            etiquetas_custom[i]: float(prediccion[0][i])
            for i in range(len(etiquetas_custom))
        }
        
        # Ordenar por probabilidad
        resultados = dict(
            sorted(resultados.items(), key=lambda x: x[1], reverse=True)
        )
        
        return resultados
    
    except Exception as e:
        print(f"Error: {e}")
        return {"Error": 1.0}

def clasificar_con_clip(imagen):
    """
    Clasifica con CLIP.
    """
    if imagen is None:
        return {"Error": 1.0}
    
    try:
        resultados = modelo_clip(imagen, candidate_labels=CATEGORIAS_CLIP)
        return {
            r['label']: float(r['score']) for r in resultados
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"Error": 1.0}

# ============================================
# INTERFAZ GRADIO MEJORADA
# ============================================

import datetime

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    
    # ENCABEZADO
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>♻️ Clasificador de Residuos con IA</h1>
        <p>
            Compara el rendimiento entre un <b>modelo personalizado</b> 
            entrenado con Teachable Machine y un <b>modelo preentrenado (CLIP)</b>.
        </p>
    </div>
    """)
    
    # FILA PRINCIPAL: IMAGEN + BOTONES
    with gr.Row():
        with gr.Column(scale=2):
            imagen_input = gr.Image(
                type="pil",
                label="📸 Subí una imagen o usá la cámara",
                sources=["upload", "webcam"]
            )

            # Ejemplos de imágenes
            gr.Markdown("#### 🧩 Ejemplos de prueba")
            ejemplos = [
                ["ejemplos/vidrio.jpeg"],
                ["ejemplos/metal.jpeg"],
                ["ejemplos/papel.jpeg"],
                ["ejemplos/plastico.jpeg"],
                ["ejemplos/organico.jpeg"],
                ["ejemplos/carton.jpeg"]
            ]
            gr.Examples(examples=ejemplos, inputs=imagen_input)
            
            with gr.Row():
                boton_custom = gr.Button("🔬 Clasificar con Modelo Personalizado", variant="primary")
                boton_clip = gr.Button("🌍 Clasificar con CLIP (OpenAI)", variant="secondary")

        # RESULTADOS LADO DERECHO
        with gr.Column(scale=1):
            resultado_custom = gr.Label(
                label="🔬 Resultado: Modelo Personalizado",
                num_top_classes=len(etiquetas_custom)
            )
            resultado_clip = gr.Label(
                label="🌍 Resultado: CLIP (Preentrenado)",
                num_top_classes=len(CATEGORIAS_CLIP)
            )

            # Indicador de estado
            estado = gr.Textbox(label="🧠 Estado del sistema", value="Listo para clasificar", interactive=False)

    # PIE DE PÁGINA
    gr.Markdown(f"""
    ---
    <div style="text-align: center; font-size: 14px; color: gray;">
        <p>📚 Desarrollado para <b>Procesamiento Digital de Imágenes y Visión por Computadora</b> — {datetime.date.today().year}</p>
        <p>Autor: <b>Alejandro Schiariti</b></p>
    </div>
    """)

    # Eventos
    def actualizar_estado(nombre_modelo):
        return f"Ejecutando inferencia con {nombre_modelo}..."

    boton_custom.click(
        fn=lambda img: clasificar_con_custom(img),
        inputs=imagen_input,
        outputs=resultado_custom
    ).then(fn=lambda _: "✅ Clasificación completada con Modelo Personalizado", inputs=None, outputs=estado)

    boton_clip.click(
        fn=lambda img: clasificar_con_clip(img),
        inputs=imagen_input,
        outputs=resultado_clip
    ).then(fn=lambda _: "✅ Clasificación completada con CLIP", inputs=None, outputs=estado)

# ============================================
# LANZAR APLICACIÓN
# ============================================

if __name__ == "__main__":
    demo.launch()
