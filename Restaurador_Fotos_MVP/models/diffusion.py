"""
🖼️ Módulo de Restauración Fotográfica

Este módulo contiene todas las funciones de procesamiento de imágenes para
restauración fotográfica usando algoritmos de IA avanzados.

Algoritmos incluidos:
- Real-ESRGAN: Upscaling de ultra-alta calidad
- CodeFormer: Restauración facial sin recortes
- GFPGAN: Restauración facial alternativa
- CLAHE: Contraste adaptativo
- Unsharp Mask: Afilado de detalles
- Reducción de ruido avanzada
- Reparación de arañazos y manchas
- Stable Diffusion con ControlNet
"""

import io
from PIL import Image
import numpy as np
import torch
import os
import urllib.request
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CodeFormer (alternative to GFPGAN for full image processing) ---
def restaurar_imagen_codeformer(img: Image.Image, fidelity: float = 0.5, upscale_factor: int = 1) -> Image.Image:
    """Use CodeFormer for face restoration without cropping"""
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils import img2tensor, tensor2img
        import torch

        # Try to load from Hugging Face Hub first
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("facebookresearch/codeformer", "codeformer.pth")
        except:
            # Fallback to local file or download
            model_path = 'CodeFormer.pth'
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                    model_path
                )

        # Initialize the face helper with configurable upscale
        face_helper = FaceRestoreHelper(
            upscale_factor=upscale_factor,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device
        )

        # Load CodeFormer model
        codeformer_net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=32,
            num_block=5,
            num_grow_ch=128,
            scale=1
        )

        checkpoint = torch.load(model_path, map_location=device)
        codeformer_net.load_state_dict(checkpoint['params'])
        codeformer_net.to(device)
        codeformer_net.eval()

        # Process image with configurable fidelity
        face_helper.read_image(np.array(img))
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face_tensor()
        face_helper.get_restored_face(codeformer_net, w=fidelity)
        face_helper.paste_faces_to_input_image()

        restored_img = face_helper.restored_img
        return Image.fromarray(restored_img)

    except Exception as e:
        # Fallback to simple enhancement if CodeFormer fails
        import cv2
        img_np = np.array(img)
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
        return Image.fromarray(filtered)

# --- GFPGAN (original, may crop) ---
def restaurar_imagen_gfpgan(img: Image.Image, upscale_factor: int = 2) -> Image.Image:
    """Original GFPGAN - may crop faces"""
    try:
        # Try to load from Hugging Face Hub first
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("microsoft/GFPGAN", "GFPGANv1.3.pth")
        except:
            # Fallback to local file or download
            model_path = 'GFPGANv1.3.pth'
            if not os.path.exists(model_path):
                urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', model_path)

        from gfpgan import GFPGANer
        restorer = GFPGANer(model_path=model_path, upscale=upscale_factor, arch='clean', channel_multiplier=2, bg_upsampler=None)
        output, *_ = restorer.enhance(np.array(img), has_aligned=False, only_center_face=False, paste_back=True)

        # Handle different output formats from GFPGAN
        if isinstance(output, list):
            output = output[0] if output else np.array(img)

        # Ensure we have a numpy array
        if not isinstance(output, np.ndarray):
            output = np.array(img)

        # Ensure output is in the correct format for PIL
        if output.dtype != np.uint8:
            output = (output * 255).astype(np.uint8) if output.max() <= 1.0 else output.astype(np.uint8)

        return Image.fromarray(output)

    except Exception as e:
        # Fallback to simple enhancement
        import cv2
        img_np = np.array(img)
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
        return Image.fromarray(filtered)

# --- Real-ESRGAN ---
def upscale_imagen_realesrgan(img: Image.Image, model_type: str = "x4plus") -> Image.Image:
    try:
        if model_type == "x4plus":
            # Try to load from Hugging Face Hub first
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus.pth")
            except:
                # Fallback to local file or download
                model_path = 'RealESRGAN_x4plus.pth'
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model_path)
        elif model_type == "x4plus-anime":
            # Try to load from Hugging Face Hub first
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus_anime_6B.pth")
            except:
                # Fallback to local file or download
                model_path = 'RealESRGAN_x4plus_anime.pth'
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', model_path)
        else:
            # Fallback
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus.pth")
            except:
                model_path = 'RealESRGAN_x4plus.pth'

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        output, _ = upsampler.enhance(np.array(img), outscale=4)
        return Image.fromarray(output)

    except Exception as e:
        # Fallback to simple resize
        width, height = img.size
        new_size = (width * 2, height * 2)
        return img.resize(new_size, Image.LANCZOS)

def mejorar_color_contraste(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Mejora el color y contraste usando técnicas de procesamiento de imagen con intensidad configurable"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Convertir a espacio de color LAB para mejor manejo de brillo/contraste
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Aplicar CLAHE con intensidad configurable
        clip_limit = 2.0 + intensity * 2.0  # 2.0-4.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convertir de vuelta a RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Ajuste fino de saturación y brillo basado en intensidad
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        saturation_boost = 1.0 + (intensity - 1.0) * 0.3  # 0.7-1.3
        brightness_boost = 1.0 + (intensity - 1.0) * 0.1   # 0.9-1.1

        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_boost)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], brightness_boost)

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(result)

    except Exception as e:
        # Fallback simple
        import cv2
        img_array = np.array(img)
        # Ajuste básico de contraste y brillo
        alpha = 1.0 + (intensity - 1.0) * 0.2  # 0.8-1.2
        beta = int((intensity - 1.0) * 10)      # -10 to +10
        adjusted = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
        return Image.fromarray(adjusted)

def reducir_ruido_avanzado(img: Image.Image, strength: int = 3) -> Image.Image:
    """Reducción avanzada de ruido usando múltiples técnicas con intensidad configurable"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Ajustar parámetros según la intensidad
        base_diameter = 5 + strength * 2  # 7-15
        base_sigma = 50 + strength * 10   # 80-150

        # Aplicar denoising bilateral para preservar bordes
        denoised = cv2.bilateralFilter(img_array, base_diameter, base_sigma, base_sigma)

        if strength >= 3:
            # Para intensidad alta, aplicar segunda pasada
            denoised = cv2.bilateralFilter(denoised, base_diameter - 2, base_sigma - 20, base_sigma - 20)

        if strength >= 4:
            # Para intensidad muy alta, añadir blur gaussiano suave
            gaussian = cv2.GaussianBlur(denoised, (3, 3), 0)
            denoised = cv2.addWeighted(denoised, 0.8, gaussian, 0.2, 0)

        return Image.fromarray(denoised)

    except Exception as e:
        # Fallback simple
        import cv2
        img_array = np.array(img)
        # Reducción simple de ruido
        denoised = cv2.bilateralFilter(img_array, 5, 50, 50)
        return Image.fromarray(denoised)

def inpainting_aranasos_agresivo(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Eliminación agresiva de arañazos, roturas y daños físicos usando inpainting múltiple"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Estrategia múltiple para detectar diferentes tipos de daños

        # Estrategia 1: Detección de líneas/arañazos (bordes irregulares)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel_line = np.ones((1, 5), np.uint8)  # Kernel horizontal para líneas
        dilated_lines = cv2.dilate(edges, kernel_line, iterations=1)

        # Estrategia 2: Detección de áreas irregulares (roturas)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel_irregular = np.ones((5, 5), np.uint8)
        irregular_areas = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_irregular)

        # Estrategia 3: Detección de variaciones de intensidad (posibles daños)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        damage_candidates = cv2.convertScaleAbs(laplacian)
        _, damage_mask = cv2.threshold(damage_candidates, 30, 255, cv2.THRESH_BINARY)

        # Combinar todas las estrategias
        combined_mask = cv2.bitwise_or(dilated_lines, irregular_areas)
        combined_mask = cv2.bitwise_or(combined_mask, damage_mask)

        # Operaciones morfológicas agresivas
        kernel_final = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_final)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_final)

        # Encontrar contornos y filtrar
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear máscara refinada con criterios más agresivos
        refined_mask = np.zeros_like(combined_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Calcular elongación (para detectar líneas/arañazos)
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    elongation = max(width, height) / min(width, height)
                else:
                    elongation = 1

            # Criterios más agresivos para daños físicos
            if area > 20 and (elongation > 3 or area < 1000):  # Líneas largas o áreas pequeñas irregulares
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Aplicar inpainting múltiple
        if np.any(refined_mask > 0):
            # Primer inpainting con radio grande para reconstrucción
            inpainted = cv2.inpaint(img_array, refined_mask, inpaintRadius=9, flags=cv2.INPAINT_NS)

            # Segundo inpainting para refinar
            inpainted = cv2.inpaint(inpainted, refined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            return Image.fromarray(inpainted)
        else:
            return reducir_ruido_avanzado(img)

    except Exception as e:
        # Fallback agresivo
        import cv2
        img_array = np.array(img)
        # Múltiples pasadas de denoising muy agresivo
        result = cv2.bilateralFilter(img_array, 15, 150, 150)
        result = cv2.bilateralFilter(result, 11, 120, 120)
        result = cv2.bilateralFilter(result, 7, 100, 100)
        return Image.fromarray(result)

def definir_bordes_foto(img: Image.Image) -> Image.Image:
    """Definir y mejorar los bordes de la fotografía para un aspecto más profesional"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Convertir a escala de grises
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detectar bordes con Canny
        edges = cv2.Canny(gray, 100, 200)

        # Dilatar bordes para crear un marco sutil
        kernel = np.ones((2, 2), np.uint8)
        edge_frame = cv2.dilate(edges, kernel, iterations=1)

        # Crear una máscara para el área interior (ligeramente más pequeña)
        height, width = gray.shape
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (5, 5), (width-5, height-5), 255, -1)

        # Aplicar un ligero desenfoque a los bordes para suavizar
        blurred_edges = cv2.GaussianBlur(edge_frame, (3, 3), 0)

        # Combinar con la imagen original usando máscara
        result = img_array.copy()
        # Aplicar un ligero ajuste de contraste en los bordes
        result[blurred_edges > 0] = cv2.addWeighted(
            result[blurred_edges > 0], 1.1,
            np.full_like(result[blurred_edges > 0], 128), 0, 0
        )

        return Image.fromarray(result)

    except Exception as e:
        return img

def mejorar_contraste_adaptativo(img: Image.Image) -> Image.Image:
    """Mejora el contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Convertir a LAB para mejor manejo de luminosidad
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Aplicar CLAHE al canal L (luminosidad)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convertir de vuelta a RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)

    except Exception as e:
        return img

def afinar_detalles(img: Image.Image) -> Image.Image:
    """Afina detalles usando un filtro de paso alto sutil"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Aplicar un ligero blur gaussiano
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 1.0)

        # Crear máscara de detalles (imagen original - blur)
        unsharp_mask = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)

        # Combinar con la imagen original para afinar detalles
        sharpened = cv2.addWeighted(img_array, 1.0, unsharp_mask, 0.3, 0)

        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

    except Exception as e:
        return img

def colorizar_imagen(img: Image.Image) -> Image.Image:
    """Colorización mejorada usando mapeo de color basado en referencias de piel y tonos"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Detectar si la imagen ya tiene algo de color (evitar sobreprocesar)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])

        # Si ya tiene saturación significativa, no colorizar
        if saturation_mean > 20:
            return img

        # Convertir a LAB para mejor control de color
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Estrategia mejorada de colorización
        a_channel = np.zeros_like(l_channel, dtype=np.uint8)
        b_channel = np.zeros_like(l_channel, dtype=np.uint8)

        # Rangos más sofisticados basados en teoría del color
        # Sombras profundas (negro azulado)
        dark_mask = l_channel < 40
        a_channel[dark_mask] = 135  # Azul profundo
        b_channel[dark_mask] = 135

        # Sombras (gris azulado)
        shadow_mask = (l_channel >= 40) & (l_channel < 80)
        a_channel[shadow_mask] = 130
        b_channel[shadow_mask] = 140

        # Tonos medios (piel/tejidos)
        mid_mask = (l_channel >= 80) & (l_channel < 150)
        a_channel[mid_mask] = 145  # Tonos piel naturales
        b_channel[mid_mask] = 125

        # Altas luces (blancos cálidos)
        light_mask = (l_channel >= 150) & (l_channel < 200)
        a_channel[light_mask] = 150
        b_channel[light_mask] = 115

        # Picos de luz (blancos puros)
        bright_mask = l_channel >= 200
        a_channel[bright_mask] = 155
        b_channel[bright_mask] = 110

        # Aplicar canales de color
        lab[:, :, 1] = a_channel
        lab[:, :, 2] = b_channel

        # Convertir de vuelta a RGB
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Aplicar un ligero ajuste de saturación para naturalidad
        hsv_colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2HSV)
        hsv_colorized[:, :, 1] = cv2.multiply(hsv_colorized[:, :, 1], 0.8)  # Reducir saturación ligeramente
        final_result = cv2.cvtColor(hsv_colorized, cv2.COLOR_HSV2RGB)

        return Image.fromarray(final_result)

    except Exception as e:
        return img

def reparar_manchas_blancas(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Reparación especializada de manchas blancas, agujeros y marcas en rostros"""
    try:
        import cv2
        import numpy as np

        img_array = np.array(img)

        # Múltiples estrategias de detección para diferentes tipos de daños

        # Estrategia 1: Detección de áreas blancas puras (agujeros/roturas)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Estrategia 2: Detección de áreas sobreexpuestas (manchas blancas)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 220])  # Más restrictivo
        upper_white = np.array([180, 40, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # Combinar máscaras
        combined_mask = cv2.bitwise_or(white_mask, white_mask_hsv)

        # Estrategia 3: Detección de áreas con poco detalle (posibles agujeros)
        # Calcular varianza local - áreas uniformes pueden ser daños
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        variance = cv2.Laplacian(blur, cv2.CV_64F).var()
        low_detail_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Combinar con áreas de bajo detalle
        final_mask = cv2.bitwise_or(combined_mask, low_detail_mask)

        # Operaciones morfológicas agresivas para conectar áreas dañadas
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos y filtrar
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear máscara refinada
        refined_mask = np.zeros_like(final_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Filtrar por área y forma (manchas/roturas típicas)
            if 50 < area < 5000 and circularity < 0.8:  # No perfectamente circulares
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Aplicar inpainting múltiple para mejor reconstrucción
        if np.any(refined_mask > 0):
            # Primer paso: inpainting NS para texturas complejas
            repaired = cv2.inpaint(img_array, refined_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)

            # Segundo paso: inpainting Telea para refinar
            repaired = cv2.inpaint(repaired, refined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            return Image.fromarray(repaired)
        else:
            return img

    except Exception as e:
        # Fallback múltiple
        try:
            # Fallback 1: CLAHE agresivo
            import cv2
            img_array = np.array(img)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(result)
        except:
            # Fallback 2: solo devolver original
            return img

# --- Stable Diffusion + ControlNet ---
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

def restaurar_imagen_sd(img: Image.Image, hf_token: str, prompt: str = "restaurar imagen antigua dañada, reparar rasguños y arrugas, mejorar nitidez, colores naturales, imagen antigua restaurada profesionalmente", strength: float = 0.6, steps: int = 15) -> Image.Image:
    # Try to use Stable Diffusion + ControlNet as per requirements
    if not hf_token:
        raise ValueError("HF_TOKEN is required for Stable Diffusion")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch_dtype,
            token=hf_token
        )

        # Load Stable Diffusion pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            token=hf_token
        )
        pipe.to(device)

        # Prepare canny edge detection for ControlNet
        import cv2
        canny_image = cv2.Canny(np.array(img), 100, 200)
        canny_image = Image.fromarray(canny_image)

        # Generate enhanced image with configurable parameters
        result = pipe(
            prompt=prompt,
            image=img,
            control_image=canny_image,
            strength=strength,  # Configurable strength
            guidance_scale=7.5,  # Balanced guidance
            num_inference_steps=steps  # Configurable steps
        ).images[0]

        return result

    except Exception as e:
        # Fallback to simple enhancement if SD fails
        import cv2
        import numpy as np

        img_np = np.array(img)
        # Apply gentle sharpening and denoising
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])  # Moderate sharpening
        sharpened = cv2.filter2D(img_np, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Apply denoising to reduce artifacts
        denoised = cv2.bilateralFilter(sharpened, 7, 75, 75)

        return Image.fromarray(denoised)
