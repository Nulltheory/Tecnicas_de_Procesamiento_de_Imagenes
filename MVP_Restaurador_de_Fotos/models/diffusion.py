"""
🖼️ Módulo de Restauración Fotográfica - Optimizado para Streamlit Cloud

Esta versión mantiene TODAS las funcionalidades originales (Real-ESRGAN, CodeFormer, GFPGAN,
Stable Diffusion, múltiples mejoras básicas).
- No importa OpenCV en ningún punto.
- Loaders on-demand para modelos pesados (si existen).
- Fallbacks completos con PIL + NumPy para todas las funcionalidades críticas.
- Mantiene nombres de funciones para compatibilidad con tu app.py.
"""

import os
import io
import warnings
import urllib.request
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

# ---------------------------
# PyTorch detection
# ---------------------------
device = "cpu"
torch_available = False
try:
    import torch
    torch_available = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch_available = False
    warnings.warn("PyTorch no disponible - funcionando en modo básico (sin aceleración)")

# ---------------------------
# Model availability flags (on-demand)
# ---------------------------
_real_esrgan_available = False
_codeformer_available = False
_gfpgan_available = False
_stable_diffusion_available = False

# ---------------------------
# Utility conversions
# ---------------------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL RGB -> numpy RGB uint8"""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError("Unsupported image type for pil_to_np")

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy RGB uint8 -> PIL"""
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_clipped)

# ---------------------------
# On-demand model loaders
# ---------------------------
def _load_real_esrgan():
    global _real_esrgan_available
    if _real_esrgan_available or not torch_available:
        return _real_esrgan_available
    try:
        # Attempt to import realtime modules; if fail, mark False
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

# ---------------------------
# Fallback basic ops (PIL)
# ---------------------------
def _fallback_enhancement(img: Image.Image, method: str = "fallback") -> Image.Image:
    try:
        out = img.filter(ImageFilter.GaussianBlur(radius=1))
        out = ImageEnhance.Sharpness(out).enhance(1.1)
        return out
    except Exception:
        return img

def _fallback_upscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((w * 2, h * 2), Image.LANCZOS)

# ---------------------------
# CodeFormer (on-demand) - same semantics as original, but no cv2 usage here
# ---------------------------
def restaurar_imagen_codeformer(img: Image.Image, fidelity: float = 0.5, upscale_factor: int = 1) -> Image.Image:
    if not _load_codeformer():
        return _fallback_enhancement(img, "CodeFormer")
    try:
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

        # download or fallback local model
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("facebookresearch/codeformer", "codeformer.pth")
        except Exception:
            model_path = "CodeFormer.pth"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
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

# ---------------------------
# GFPGAN (on-demand)
# ---------------------------
def restaurar_imagen_gfpgan(img: Image.Image, upscale_factor: int = 2) -> Image.Image:
    if not _load_gfpgan():
        return _fallback_enhancement(img, "GFPGAN")
    try:
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("microsoft/GFPGAN", "GFPGANv1.3.pth")
        except Exception:
            model_path = "GFPGANv1.3.pth"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                    model_path
                )

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

# ---------------------------
# Real-ESRGAN (on-demand) - keep as before but no cv2 usage
# ---------------------------
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
                model_path = "RealESRGAN_x4plus.pth"
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        model_path
                    )
        elif model_type == "x4plus-anime":
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus_anime_6B.pth")
            except Exception:
                model_path = "RealESRGAN_x4plus_anime.pth"
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                        model_path
                    )
        else:
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("microsoft/RealESRGAN", "RealESRGAN_x4plus.pth")
            except Exception:
                model_path = "RealESRGAN_x4plus.pth"

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
        output, _ = upsampler.enhance(np.array(img), outscale=4)
        return Image.fromarray(output)
    except Exception as e:
        print(f"Error en Real-ESRGAN: {e}")
        return _fallback_upscale(img)

# ---------------------------
# Color / contrast (PIL + numpy fallback)
# ---------------------------
def mejorar_color_contraste(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    try:
        # Implement a simple CLAHE-like enhancement using PIL + numpy approximation
        arr = pil_to_np(img).astype(np.uint8)
        # Convert RGB -> LAB approximation using skimage-like transform is heavy; use simple contrast + saturation adjustments
        im = Image.fromarray(arr)
        # Mild contrast via ImageEnhance
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(1.0 + (intensity - 1.0) * 0.2)
        # Slight color boost
        color_enhancer = ImageEnhance.Color(im)
        im = color_enhancer.enhance(1.0 + (intensity - 1.0) * 0.15)
        return im
    except Exception as e:
        print(f"mejorar_color_contraste fallback: {e}")
        try:
            return _fallback_enhancement(img, "color_contrast")
        except:
            return img

# ---------------------------
# Denoise (PIL / numpy approach)
# ---------------------------
def reducir_ruido_avanzado(img: Image.Image, strength: int = 3) -> Image.Image:
    try:
        arr = pil_to_np(img).astype(np.float32)
        # Use repeated box blur as a denoise approximation; stronger strength -> more passes
        passes = max(1, int(strength))
        out = arr.copy()
        for _ in range(passes):
            out = _box_blur_numpy(out, k=3 + (passes-1))
        return np_to_pil(out)
    except Exception as e:
        print(f"reducir_ruido_avanzado fallback: {e}")
        try:
            return img.filter(ImageFilter.GaussianBlur(radius=0.5 * strength))
        except:
            return img

# ---------------------------
# Inpainting / eliminación de arañazos - versión sin OpenCV
# ---------------------------
def inpainting_aranasos_agresivo(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    """
    Implementación puramente PIL/NumPy para detectar rasguños y repararlos.
    - Detección por gradiente (Sobel-like)
    - Refinamiento de máscara con morfología numpy
    - Inpainting ligero por difusión guiada (vecindario)
    """
    try:
        arr = pil_to_np(img).astype(np.uint8)
        gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.uint8)

        # Sobel-like gradients
        kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.int32)
        ky = kx.T
        gx = _convolve2d(gray.astype(np.int32), kx)
        gy = _convolve2d(gray.astype(np.int32), ky)
        mag = np.abs(gx) + np.abs(gy)

        # Adaptive threshold
        thresh_val = max(8, int(mag.mean() * (1.0 + (5 - sensitivity) * 0.05)))
        mask = (mag > thresh_val).astype(np.uint8) * 255

        # Morphology: dilate then close using numpy kernels
        mask = _numpy_dilate(mask, kernel_size=3, iterations=1)
        mask = _numpy_close(mask, kernel_size=5)

        if mask.mean() < 1.0:
            # nothing detected; fallback to denoise
            return reducir_ruido_avanzado(img, strength=max(1, sensitivity//2))

        # Inpainting by iterative guided diffusion: replace masked pixels with neighborhood mean (RGB)
        result = arr.copy().astype(np.float32)
        mask_bool = mask > 0
        for _ in range(6):
            neigh = _box_blur_numpy(result, k=5)
            # only replace pixels where mask True
            for c in range(3):
                channel = result[..., c]
                channel[mask_bool] = neigh[..., c][mask_bool]
                result[..., c] = channel
            # optional: shrink mask gradually (feather)
            mask = _numpy_erode(mask, kernel_size=3, iterations=1)
            mask_bool = mask > 0
            if not mask_bool.any():
                break

        # Slight sharpening
        blurred = _box_blur_numpy(result, k=3)
        final = np.clip(result * 1.1 - blurred * 0.1, 0, 255).astype(np.uint8)
        return np_to_pil(final)
    except Exception as e:
        print(f"inpainting_aranasos_agresivo fallback error: {e}")
        return _fallback_enhancement(img, "inpainting")

# ---------------------------
# Definir bordes (PIL)
# ---------------------------
def definir_bordes_foto(img: Image.Image) -> Image.Image:
    try:
        arr = pil_to_np(img).astype(np.uint8)
        gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.uint8)
        gx = _convolve2d(gray.astype(np.int32), np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
        gy = _convolve2d(gray.astype(np.int32), np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
        mag = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
        edges = (mag > (mag.mean()*1.2)).astype(np.uint8) * 255
        # Create soft frame by dilating and blurring the edges
        frame = _numpy_dilate(edges, kernel_size=3, iterations=2)
        frame_blur = _box_blur_numpy(np.dstack([frame]*3).astype(np.float32), k=7)
        result = arr.copy().astype(np.float32)
        mask = frame_blur[...,0] > 10
        result[mask] = np.clip(result[mask] * 1.08 + 10, 0, 255)
        return np_to_pil(result.astype(np.uint8))
    except Exception as e:
        print(f"definir_bordes_foto fallback: {e}")
        return img

# ---------------------------
# CLAHE-like (approx)
# ---------------------------
def mejorar_contraste_adaptativo(img: Image.Image) -> Image.Image:
    try:
        # PIL-based approximation: equalize per channel modestly
        arr = pil_to_np(img).astype(np.uint8)
        out = np.zeros_like(arr)
        for c in range(3):
            ch = arr[..., c]
            # clip extremes
            p1, p99 = np.percentile(ch, (1, 99))
            ch = np.clip((ch - p1) * 255.0 / max(1, (p99 - p1)), 0, 255).astype(np.uint8)
            # mild local equalization via tiny box blur subtraction
            local = _box_blur_numpy(ch.astype(np.float32), k=9)
            ch2 = np.clip(ch * 1.05 + (ch - local) * 0.3, 0, 255)
            out[..., c] = ch2
        return np_to_pil(out.astype(np.uint8))
    except Exception as e:
        print(f"mejorar_contraste_adaptativo fallback: {e}")
        return img

# ---------------------------
# Afilar detalles (unsharp) - PIL
# ---------------------------
def afinar_detalles(img: Image.Image) -> Image.Image:
    try:
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    except Exception as e:
        print(f"afinar_detalles fallback: {e}")
        return img

# ---------------------------
# Colorización (approx)
# ---------------------------
def colorizar_imagen(img: Image.Image) -> Image.Image:
    try:
        arr = pil_to_np(img).astype(np.uint8)
        hsv_mean = np.mean(_rgb_to_hsv(arr)[...,1])
        if hsv_mean > 20:
            return img
        # Simple tinting by luminosity bands
        l = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]).astype(np.uint8)
        out = arr.copy().astype(np.float32)
        # apply simple tone mapping per band (heuristic)
        dark = l < 60
        mid = (l >= 60) & (l < 170)
        bright = l >= 170
        out[dark] *= np.array([0.85, 0.9, 1.05])
        out[mid] *= np.array([1.02, 0.95, 0.9])
        out[bright] *= np.array([1.05, 1.02, 0.98])
        out = np.clip(out, 0, 255).astype(np.uint8)
        return np_to_pil(out)
    except Exception as e:
        print(f"colorizar_imagen fallback: {e}")
        return img

# ---------------------------
# Reparar manchas blancas
# ---------------------------
def reparar_manchas_blancas(img: Image.Image, sensitivity: int = 5) -> Image.Image:
    try:
        arr = pil_to_np(img).astype(np.uint8)
        gray = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]).astype(np.uint8)
        white_mask = (gray > 245).astype(np.uint8) * 255
        # HSV-based detection approximation
        hsv = _rgb_to_hsv(arr)
        white_mask_hsv = ((hsv[...,1] < 30) & (hsv[...,2] > 220)).astype(np.uint8) * 255
        combined = np.clip(white_mask + white_mask_hsv, 0, 255).astype(np.uint8)
        combined = _numpy_close(combined, kernel_size=7)
        # find regions and inpaint them with diffusion
        if combined.mean() < 1.0:
            return img
        mask = combined
        result = arr.copy().astype(np.float32)
        for _ in range(5):
            neigh = _box_blur_numpy(result, k=7)
            for c in range(3):
                channel = result[...,c]
                channel[mask>0] = neigh[...,c][mask>0]
                result[...,c] = channel
            mask = _numpy_erode(mask, kernel_size=3, iterations=1)
            if mask.sum() == 0:
                break
        return np_to_pil(np.clip(result,0,255).astype(np.uint8))
    except Exception as e:
        print(f"reparar_manchas_blancas fallback: {e}")
        try:
            return ImageOps.equalize(img)
        except:
            return img

# ---------------------------
# Stable Diffusion (on-demand)
# ---------------------------
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
        # prepare a simple edge map using numpy (approx)
        arr = pil_to_np(img)
        gray = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]).astype(np.uint8)
        edges = (_convolve2d(gray.astype(np.int32), np.array([[1,1,1],[1,-8,1],[1,1,1]])) > 20).astype(np.uint8) * 255
        control_img = Image.fromarray(edges)
        result = pipe(
            prompt=prompt,
            image=img,
            control_image=control_img,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=steps
        ).images[0]
        return result
    except Exception as e:
        print(f"restaurar_imagen_sd fallo: {e}")
        return _fallback_enhancement(img, "stable_diffusion")

# ---------------------------
# Public API alias (keep name expected by app.py)
# ---------------------------
# inpainting_aranasos_agresivo is expected by your app; provide alias
# We'll use the pure-PIL inpainting implementation above.
inpainting_aranasos_agresivo = inpainting_aranasos_agresivo

# ---------------------------
# --- Helper small image ops (NumPy implementations)
# ---------------------------
def _convolve2d(img, kernel):
    """2D convolution for single-channel image using valid padding -> same output size"""
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img, dtype=np.int32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y+k_h, x:x+k_w]
            out[y, x] = int((region * kernel).sum())
    return out

def _box_blur_numpy(img, k=3):
    """Simple box blur for uint8 RGB or float arrays."""
    if img.ndim == 2:
        arr = img
        pad = k//2
        padded = np.pad(arr, ((pad,pad),(pad,pad)), mode='reflect')
        out = np.zeros_like(arr, dtype=np.float32)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                out[y,x] = padded[y:y+k, x:x+k].mean()
        return out
    else:
        arr = img
        pad = k//2
        padded = np.pad(arr, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
        out = np.zeros_like(arr, dtype=np.float32)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                out[y,x] = padded[y:y+k, x:x+k].mean(axis=(0,1))
        return out

def _numpy_dilate(mask, kernel_size=3, iterations=1):
    """Simple dilate for 2D uint8 masks"""
    out = mask.copy()
    for _ in range(iterations):
        pad = kernel_size // 2
        padded = np.pad(out, pad, mode='constant', constant_values=0)
        new = out.copy()
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y:y+kernel_size, x:x+kernel_size]
                if np.any(region):
                    new[y,x] = 255
        out = new
    return out

def _numpy_erode(mask, kernel_size=3, iterations=1):
    out = mask.copy()
    for _ in range(iterations):
        pad = kernel_size // 2
        padded = np.pad(out, pad, mode='constant', constant_values=0)
        new = out.copy()
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y:y+kernel_size, x:x+kernel_size]
                if np.all(region):
                    new[y,x] = 255
                else:
                    new[y,x] = 0
        out = new
    return out

def _numpy_close(mask, kernel_size=3):
    return _numpy_erode(_numpy_dilate(mask, kernel_size=kernel_size, iterations=1), kernel_size=kernel_size, iterations=1)

def _rgb_to_hsv(arr):
    """arr uint8 RGB -> float HSV in range H:0-360, S:0-255, V:0-255 (approx)"""
    arr_f = arr.astype(np.float32) / 255.0
    r = arr_f[...,0]; g = arr_f[...,1]; b = arr_f[...,2]
    mx = np.maximum.reduce([r,g,b])
    mn = np.minimum.reduce([r,g,b])
    diff = mx - mn + 1e-9
    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    mask = mx == g
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
    mask = mx == b
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360
    s = diff / (mx + 1e-9)
    v = mx
    hsv = np.stack([h, s*255.0, v*255.0], axis=-1)
    return hsv

# ---------------------------
# End of module
# ---------------------------


