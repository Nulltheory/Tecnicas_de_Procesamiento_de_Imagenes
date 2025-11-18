"""
🖼️ Módulo de Restauración Fotográfica - Optimizado para Streamlit Cloud

Esta versión mantiene TODAS las funcionalidades originales (Real-ESRGAN, CodeFormer, GFPGAN,
Stable Diffusion, múltiples mejoras básicas) y añade:
- Sistema híbrido robusto para la Fase 2 (Reparación Estructural): usa OpenCV cuando
  está "saludable" y un fallback numpy/PIL cuando no lo está.
- Detección explícita del estado de OpenCV y manejo de errores relacionados con libGL/libgomp.

Instrucciones:
- Este archivo asume que las dependencias del sistema (packages.txt) incluyen libgl1-mesa-glx, libgomp1, etc.
- Las librerías pesadas se cargan *on-demand* (evitan fallos en entornos limitados).
"""

import io
import os
import warnings
import urllib.request
from PIL import Image
import numpy as np

# ------------------------------------------------------------
# Detectar PyTorch y dispositivo
# ------------------------------------------------------------

device = "cpu"
torch_available = False
try:
    import torch
    torch_available = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch_available = False
    warnings.warn("PyTorch no disponible - funcionando en modo básico")

# ------------------------------------------------------------
# Flags de disponibilidad (cargados on-demand)
# ------------------------------------------------------------

_real_esrgan_available = False
_codeformer_available = False
_gfpgan_available = False
_stable_diffusion_available = False

# ------------------------------------------------------------
# Helpers de conversión PIL <-> OpenCV
# ------------------------------------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img)
        # PIL RGB -> cv2 BGR
        return arr[:, :, ::-1].copy()
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Unsupported image type")


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    if isinstance(img, np.ndarray):
        # cv2 BGR -> PIL RGB
        return Image.fromarray(img[:, :, ::-1])
    elif isinstance(img, Image.Image):
        return img
    else:
        raise TypeError("Unsupported image type")

# ------------------------------------------------------------
# Comprobación de estado de OpenCV
# ------------------------------------------------------------

def opencv_is_healthy():
    """Chequea si OpenCV está presente y con build razonable.
    Devuelve True si build information indica soporte OpenGL y OpenMP (buena heurística).
    """
    try:
        import cv2
        info = cv2.getBuildInformation()
        ok = True
        # heurística: si contiene OpenGL: YES y OpenMP: YES lo consideramos "sano"
        if "OpenGL: YES" not in info:
            ok = False
        if "OpenMP: YES" not in info and "With TBB" not in info:
            # puede no tener OpenMP pero tener TBB; si ninguno -> degradado
            ok = False
        return ok
    except Exception:
        return False

# ------------------------------------------------------------
# Loaders on-demand para modelos pesados
# ------------------------------------------------------------

def _load_real_esrgan():
    global _real_esrgan_available
    if _real_esrgan_available or not torch_available:
        return _real_esrgan_available
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from realesrgan import RealESRGANer  # type: ignore
        _real_esrgan_available = True
        return True
    except Exception as e:
        print(f"Real-ESRGAN no disponible: {e}")
        return False


def _load_codeformer():
    global _codeformer_available
    if _codeformer_available or not torch_available:
        return _codeformer_available
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        _codeformer_available = True
        return True
    except Exception as e:
        print(f"CodeFormer no disponible: {e}")
        return False


def _load_gfpgan():
    global _gfpgan_available
    if _gfpgan_available or not torch_available:
        return _gfpgan_available
    try:
        from gfpgan import GFPGANer  # type: ignore
        _gfpgan_available = True
        return True
    except Exception as e:
        print(f"GFPGAN no disponible: {e}")
        return False


def _load_stable_diffusion():
    global _stable_diffusion_available
    if _stable_diffusion_available or not torch_available:
        return _stable_diffusion_available
    try:
        from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel  # type: ignore
        _stable_diffusion_available = True
        return True
    except Exception as e:
        print(f"Stable Diffusion no disponible: {e}")
        return False

# ------------------------------------------------------------
# Fallbacks básicos (PIL) para cuando no hay soporte
# ------------------------------------------------------------

def _fallback_enhancement(img: Image.Image, method: str = "fallback") -> Image.Image:
    try:
        from PIL import ImageFilter, ImageEnhance
        result = img.filter(ImageFilter.GaussianBlur(radius=1))
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.1)
        return result
    except Exception:
        return img

def _fallback_upscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((w*2, h*2), Image.LANCZOS)

# ------------------------------------------------------------
# CodeFormer
# ------------------------------------------------------------

def restaurar_imagen_codeformer(img: Image.Image, fidelity: float = 0.5, upscale_factor: int = 1) -> Image.Image:
    if not _load_codeformer():
        return _fallback_enhancement(img, "CodeFormer")
    try:
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        # intentar descarga desde HF hub o fallback a release
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("facebookresearch/codeformer", "codeformer.pth")
        except Exception:
            model_path = 'CodeFormer.pth'
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                    model_path
                )

        face_helper = FaceRestoreHelper(
            upscale_factor=upscale_factor,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device
        )

        codeformer_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=5, num_grow_ch=128, scale=1)
        checkpoint = torch.load(model_path, map_location=device)
        codeformer_net.load_state_dict(checkpoint['params'])
        codeformer_net.to(device)
        codeformer_net.eval()

        face_helper.read_image(np.array(img))
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face_tensor()
        face_helper.get_restored_face(codeformer_net, w=fidelity)
        face_helper.paste_faces_to_input_image()
        restored_img = face_helper.restored_img
        return Image.fromarray(restored_img)
    except Exception as e:
        print(f"Error en CodeFormer: {e}")
        return _fallback_enhancement(img, "CodeFormer")

# ------------------------------------------------------------
# GFPGAN
# ------------------------------------------------------------

def restaurar_imagen_gfpgan(img: Image.Image, upscale_factor: int = 2) -> Image.Image:
    if not _load_gfpgan():
        return _fallback_enhancement(img, "GFPGAN")
    try:
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("microsoft/GFPGAN", "GFPGANv1.3.pth")
        except Exception:
            model_path = 'GFPGANv1.3.pth'
            if not os.path.exists(model_path):
                urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', model_path)

        from gfpgan import GFPGANer  # type: ignore
        restorer = GFPGANer(model_path=model_path, upscale=upscale_factor, arch='clean', channel_multiplier=2, bg_upsampler=None)
        output, *_ = restorer.enhance(np.array(img), has_aligned=False, only_center_face=False, paste_back=True)

        if isinstance(output, list):
            output = output[0] if output else np.array(img)
        if not isinstance(output, np.ndarray):
            output = np.array(img)
        if output.dtype != np.uint8:
            output = (output * 255).astype(np.uint8) if output.max() <= 1.0 else output.astype(np.uint8)
        return Image.fromarray(output)
    except Exception as e:
        print(f"Error en GFPGAN: {e}")
        return _fallback_enhancement(img, "GFPGAN")

# ------------------------------------------------------------
# Real-ESRGAN
# ------------------------------------------------------------

def upscale_imagen_realesrgan(img: Image.Image, model_type: str = "x4plus") -> Image.Image:
    if not _load_real_esrgan():
        return _fallback_upscale(img)
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from realesrgan import RealESRGANer  # type: ignore

        if model_type == "x4plus":
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus.pth")
            except Exception:
                model_path = 'RealESRGAN_x4plus.pth'
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model_path)
        elif model_type == "x4plus-anime":
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus_anime_6B.pth")
            except Exception:
                model_path = 'RealESRGAN_x4plus_anime.pth'
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', model_path)
        else:
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus.pth")
            except Exception:
                model_path = 'RealESRGAN_x4plus.pth'

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
        output, _ = upsampler.enhance(np.array(img), outscale=4)
        return Image.fromarray(output)
    except Exception as e:
        print(f"Error en Real-ESRGAN: {e}")
        return _fallback_upscale(img)

# ------------------------------------------------------------
# Mejoras de color / contraste
# ------------------------------------------------------------

def mejorar_color_contraste(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    try:
        import cv2
        cv2.setUseOptimized(False)
        cv2.ocl.setUseOpenCL(False)
        img_array = np.array(img)

        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clip_limit = 2.0 + intensity * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        saturation_boost = 1.0 + (intensity - 1.0) * 0.3
        brightness_boost = 1.0 + (intensity - 1.0) * 0.1
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_boost)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], brightness_boost)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(result)
    except Exception as e:
        try:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            adjusted = enhancer.enhance(1.0 + (intensity - 1.0) * 0.2)
            brightness_enhancer = ImageEnhance.Brightness(adjusted)
            adjusted = brightness_enhancer.enhance(1.0 + (intensity - 1.0) * 0.1)
            return adjusted
        except Exception:
            return img

# ------------------------------------------------------------
# Reducción de ruido avanzada
# ------------------------------------------------------------

def reducir_ruido_avanzado(img: Image.Image, strength: int = 3) -> Image.Image:
    try:
        import cv2
        cv2.setUseOptimized(False)
        cv2.ocl.setUseOpenCL(False)
        img_array = np.array(img)
        base_diameter = 5 + strength * 2
        base_sigma = 50 + strength * 10
        denoised = cv2.bilateralFilter(img_array, base_diameter, base_sigma, base_sigma)
        if strength >= 3:
            denoised = cv2.bilateralFilter(denoised, base_diameter - 2, base_sigma - 20, base_sigma - 20)
        if strength >= 4:
            gaussian = cv2.GaussianBlur(denoised, (3, 3), 0)
            denoised = cv2.addWeighted(denoised, 0.8, gaussian, 0.2, 0)
        return Image.fromarray(denoised)
    except Exception as e:
        try:
            from PIL import ImageFilter
            denoised = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            return denoised
        except Exception:
            return img

# ------------------------------------------------------------
# INPAINTING AGRESIVO (original)
# ------------------------------------------------------------

def inpainting_aranasos_agresivo(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Estrategia original que usa OpenCV intensivamente. Mantener, pero
    delegaremos a la versión híbrida si OpenCV no está sano.
    """
    try:
        import cv2
        cv2.setUseOptimized(False)
        cv2.ocl.setUseOpenCL(False)
        img_array = np.array(img)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel_line = np.ones((1, 5), np.uint8)
        dilated_lines = cv2.dilate(edges, kernel_line, iterations=1)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel_irregular = np.ones((5, 5), np.uint8)
        irregular_areas = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_irregular)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        damage_candidates = cv2.convertScaleAbs(laplacian)
        _, damage_mask = cv2.threshold(damage_candidates, 30, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(dilated_lines, irregular_areas)
        combined_mask = cv2.bitwise_or(combined_mask, damage_mask)

        kernel_final = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_final)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_final)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        refined_mask = np.zeros_like(combined_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    elongation = max(width, height) / min(width, height)
                else:
                    elongation = 1
            else:
                elongation = 1
            if area > 20 and (elongation > 3 or area < 1000):
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if np.any(refined_mask > 0):
            inpainted = cv2.inpaint(img_array, refined_mask, inpaintRadius=9, flags=cv2.INPAINT_NS)
            inpainted = cv2.inpaint(inpainted, refined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            return Image.fromarray(inpainted)
        else:
            return reducir_ruido_avanzado(img)
    except Exception as e:
        print(f"inpainting_aranasos_agresivo fallo: {e}")
        return _fallback_enhancement(img, "inpainting")

# ------------------------------------------------------------
# Definir bordes
# ------------------------------------------------------------

def definir_bordes_foto(img: Image.Image) -> Image.Image:
    try:
        import cv2
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edge_frame = cv2.dilate(edges, kernel, iterations=1)
        height, width = gray.shape
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (5, 5), (width-5, height-5), 255, -1)
        blurred_edges = cv2.GaussianBlur(edge_frame, (3, 3), 0)
        result = img_array.copy()
        result[blurred_edges > 0] = cv2.addWeighted(result[blurred_edges > 0], 1.1, np.full_like(result[blurred_edges > 0], 128), 0, 0)
        return Image.fromarray(result)
    except Exception as e:
        print(f"definir_bordes_foto fallo: {e}")
        return img

# ------------------------------------------------------------
# CLAHE simple
# ------------------------------------------------------------

def mejorar_contraste_adaptativo(img: Image.Image) -> Image.Image:
    try:
        import cv2
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    except Exception as e:
        print(f"mejorar_contraste_adaptativo fallo: {e}")
        return img

# ------------------------------------------------------------
# Afilar detalles
# ------------------------------------------------------------

def afinar_detalles(img: Image.Image) -> Image.Image:
    try:
        import cv2
        img_array = np.array(img)
        gaussian = cv2.GaussianBlur(img_array, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
        sharpened = cv2.addWeighted(img_array, 1.0, unsharp_mask, 0.3, 0)
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"afinar_detalles fallo: {e}")
        return img

# ------------------------------------------------------------
# Colorización
# ------------------------------------------------------------

def colorizar_imagen(img: Image.Image) -> Image.Image:
    try:
        import cv2
        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])
        if saturation_mean > 20:
            return img
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        a_channel = np.zeros_like(l_channel, dtype=np.uint8)
        b_channel = np.zeros_like(l_channel, dtype=np.uint8)
        dark_mask = l_channel < 40
        a_channel[dark_mask] = 135
        b_channel[dark_mask] = 135
        shadow_mask = (l_channel >= 40) & (l_channel < 80)
        a_channel[shadow_mask] = 130
        b_channel[shadow_mask] = 140
        mid_mask = (l_channel >= 80) & (l_channel < 150)
        a_channel[mid_mask] = 145
        b_channel[mid_mask] = 125
        light_mask = (l_channel >= 150) & (l_channel < 200)
        a_channel[light_mask] = 150
        b_channel[light_mask] = 115
        bright_mask = l_channel >= 200
        a_channel[bright_mask] = 155
        b_channel[bright_mask] = 110
        lab[:, :, 1] = a_channel
        lab[:, :, 2] = b_channel
        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        hsv_colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2HSV)
        hsv_colorized[:, :, 1] = cv2.multiply(hsv_colorized[:, :, 1], 0.8)
        final_result = cv2.cvtColor(hsv_colorized, cv2.COLOR_HSV2RGB)
        return Image.fromarray(final_result)
    except Exception as e:
        print(f"colorizar_imagen fallo: {e}")
        return img

# ------------------------------------------------------------
# Reparar manchas blancas
# ------------------------------------------------------------

def reparar_manchas_blancas(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    try:
        import cv2
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 40, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white, upper_white)
        combined_mask = cv2.bitwise_or(white_mask, white_mask_hsv)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        low_detail_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        final_mask = cv2.bitwise_or(combined_mask, low_detail_mask)
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(final_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            if 50 < area < 5000 and circularity < 0.8:
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)
        if np.any(refined_mask > 0):
            repaired = cv2.inpaint(img_array, refined_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
            repaired = cv2.inpaint(repaired, refined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return Image.fromarray(repaired)
        else:
            return img
    except Exception as e:
        print(f"reparar_manchas_blancas fallo: {e}")
        try:
            from PIL import ImageOps
            result = ImageOps.equalize(img)
            return result
        except Exception:
            return img

# ------------------------------------------------------------
# Stable Diffusion + ControlNet
# ------------------------------------------------------------

CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

def restaurar_imagen_sd(img: Image.Image, hf_token: str, prompt: str = "restaurar imagen antigua dañada, reparar rasguños y arrugas, mejorar nitidez, colores naturales, imagen antigua restaurada profesionalmente", strength: float = 0.6, steps: int = 15) -> Image.Image:
    if not hf_token:
        raise ValueError("HF_TOKEN is required for Stable Diffusion")
    if not _load_stable_diffusion():
        return _fallback_enhancement(img, "stable_diffusion")
    try:
        from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel  # type: ignore
        import torch
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32, token=hf_token)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            token=hf_token
        )
        pipe.to(device)
        import cv2
        canny_image = cv2.Canny(np.array(img), 100, 200)
        canny_image = Image.fromarray(canny_image)
        result = pipe(
            prompt=prompt,
            image=img,
            control_image=canny_image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=steps
        ).images[0]
        return result
    except Exception as e:
        print(f"restaurar_imagen_sd fallo: {e}")
        return _fallback_enhancement(img, "stable_diffusion")

# ------------------------------------------------------------
# ----------------- SOLUCIÓN HÍBRIDA PARA FASE 2 -------------
# ------------------------------------------------------------
# Implementa dos rutas: una con OpenCV "completo" y otra fallback
# ------------------------------------------------------------

try:
    import cv2
except Exception:
    cv2 = None


def _reparar_estructural_opencv(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Versión que usa OpenCV completo para detección + inpainting doble."""
    if cv2 is None:
        return _reparar_estructural_fallback(img, sensitivity)
    try:
        img_cv = pil_to_cv2(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Detección por bordes y morfología
        edges = cv2.Canny(gray, 50, 150)
        kernel_line = np.ones((1, 5), np.uint8)
        dilated_lines = cv2.dilate(edges, kernel_line, iterations=1)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel_irregular = np.ones((5, 5), np.uint8)
        irregular_areas = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_irregular)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        damage_candidates = cv2.convertScaleAbs(laplacian)
        _, damage_mask = cv2.threshold(damage_candidates, 30, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(dilated_lines, irregular_areas)
        combined_mask = cv2.bitwise_or(combined_mask, damage_mask)
        kernel_final = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_final)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_final)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(combined_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    elongation = max(width, height) / min(width, height)
                else:
                    elongation = 1
            else:
                elongation = 1
            if area > 20 and (elongation > 3 or area < 1000):
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)
        if np.any(refined_mask > 0):
            # Inpainting doble
            inpainted = cv2.inpaint(img_cv, refined_mask, inpaintRadius=9, flags=cv2.INPAINT_NS)
            inpainted = cv2.inpaint(inpainted, refined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            # Suavizado y sharpening
            blur = cv2.GaussianBlur(inpainted, (0, 0), 3)
            sharp = cv2.addWeighted(inpainted, 1.4, blur, -0.4, 0)
            return cv2_to_pil(sharp)
        else:
            return reducir_ruido_avanzado(img)
    except Exception as e:
        print(f"_reparar_estructural_opencv fallo: {e}")
        return _reparar_estructural_fallback(img, sensitivity)


def _reparar_estructural_fallback(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Fallback que no depende de funciones OpenCV avanzadas.
    Implementa detección por gradiente, morfología simple con numpy y una difusión guiada ligera.
    """
    try:
        # Convertir a arrays RGB (PIL) y trabajar con numpy
        arr = np.array(img.convert('RGB'))
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        # Sobel-like simple
        k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        gx = cv2.filter2D(gray, -1, k) if cv2 is not None else np.abs(np.convolve(gray.flatten(), k.flatten(), mode='same')).reshape(gray.shape)
        gy = cv2.filter2D(gray, -1, k.T) if cv2 is not None else gx
        mag = np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))
        thresh_val = max(10, int(mag.mean() * 1.2))
        thresh = (mag > thresh_val).astype(np.uint8) * 255
        # Morfología simple (numpy)
        kernel = np.ones((3,3), dtype=np.uint8)
        # Dilate
        dilated = _numpy_dilate(thresh, kernel, iterations=1)
        # Crear máscara boolean
        mask = dilated > 10
        result = arr.copy()
        # Difusión guiada simple: reemplazar píxeles dañados por promedio del vecindario
        for _ in range(5):
            # promedio de 5x5
            neigh = _numpy_box_filter(result, 5)
            result[mask] = neigh[mask]
        # Sharpen ligero: unsharp
        blurred = _numpy_gaussian_blur(result, sigma=2)
        final = np.clip((1.3 * result - 0.3 * blurred), 0, 255).astype(np.uint8)
        return Image.fromarray(final)
    except Exception as e:
        print(f"_reparar_estructural_fallback fallo: {e}")
        return img

# ------------------------------------------------------------
# Implementaciones numpy para morfología / blur (usadas por fallback)
# ------------------------------------------------------------

def _numpy_pad(img, pad):
    return np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')


def _numpy_box_filter(img, ksize=3):
    # simple mean filter on RGB image
    pad = ksize//2
    padded = _numpy_pad(img, pad)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            out[y,x] = padded[y:y+ksize, x:x+ksize].mean(axis=(0,1))
    return out


def _numpy_dilate(mask, kernel, iterations=1):
    # mask: 2D uint8
    out = mask.copy()
    ky, kx = kernel.shape
    pad = max(ky, kx)//2
    padded = np.pad(out, pad, mode='constant', constant_values=0)
    for _ in range(iterations):
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y:y+ky, x:x+kx]
                if np.any(region & kernel):
                    out[y,x] = 255
        padded = np.pad(out, pad, mode='constant', constant_values=0)
    return out


def _numpy_gaussian_blur(img, sigma=1.0):
    # separable approximation using repeated box filters
    if sigma <= 0:
        return img
    k = int(max(3, np.ceil(sigma*3)))
    k = k if k%2==1 else k+1
    return _numpy_box_filter(img, k)

# ------------------------------------------------------------
# Interfaz pública de Fase 2 (reemplaza inpainting_aranasos_agresivo en app)
# ------------------------------------------------------------

def reparar_estructural(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """Función pública que el app.py debe llamar en lugar de inpainting_aranasos_agresivo.
    Elige la ruta óptima según la salud de OpenCV.
    """
    try:
        if opencv_is_healthy():
            return _reparar_estructural_opencv(img, sensitivity)
        else:
            return _reparar_estructural_fallback(img, sensitivity)
    except Exception as e:
        print(f"reparar_estructural fallo: {e}")
        return _reparar_estructural_fallback(img, sensitivity)

# ------------------------------------------------------------
# Exportar nombres originales para compatibilidad con app.py
# ------------------------------------------------------------
# Mantener nombres anteriores como alias
inpainting_aranasos_agresivo = reparar_estructural

# ------------------------------------------------------------
# Fin del módulo
# ------------------------------------------------------------
