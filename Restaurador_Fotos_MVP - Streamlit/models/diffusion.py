"""
🖼️ diffusion.py
Módulo de restauración fotográfica:
Flujo completo: GFPGAN (full-image) → Real-ESRGAN → Stable Diffusion + ControlNet
"""

import os
import warnings
import urllib.request
from PIL import Image
import numpy as np

# -------------------- Configuración global --------------------
device = "cpu"
torch_available = False
try:
    import torch
    torch_available = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    warnings.warn("PyTorch no disponible - trabajando en CPU")

_model_status = {
    "gfpgan": "not_loaded",
    "realesrgan": "not_loaded",
    "stable_diffusion": "not_loaded"
}

# -------------------- Logs y estado --------------------
def log_model(model_name: str, success: bool, error_msg: str = None):
    if success:
        _model_status[model_name] = "loaded"
        print(f"[OK] {model_name.upper()} cargado correctamente")
    else:
        _model_status[model_name] = f"failed: {error_msg}"
        print(f"[ERROR] {model_name.upper()} fallo: {error_msg}")

def get_model_status():
    return _model_status.copy()

# -------------------- GFPGAN full-image --------------------
_gfpgan_available = False
def _load_gfpgan():
    global _gfpgan_available
    if _gfpgan_available or not torch_available:
        return _gfpgan_available
    try:
        from gfpgan import GFPGANer
        _gfpgan_available = True
        log_model("gfpgan", True)
        return True
    except Exception as e:
        log_model("gfpgan", False, str(e))
        return False

def restaurar_imagen_gfpgan(img: Image.Image, upscale: int = 2) -> Image.Image:
    """
    GFPGAN full-image usando Real-ESRGAN como bg_upsampler
    """
    if not _load_gfpgan():
        raise RuntimeError("GFPGAN no disponible")
    try:
        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = "GFPGANv1.3.pth"
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                model_path
            )

        # Real-ESRGAN upsampler para fondos
        realesrgan_model_path = "RealESRGAN_x4plus.pth"
        if not os.path.exists(realesrgan_model_path):
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                realesrgan_model_path
            )

        bg_upsampler = RealESRGANer(
            scale=4,
            model_path=realesrgan_model_path,
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler
        )

        output, *_ = restorer.enhance(
            np.array(img),
            has_aligned=False,
            only_center_face=False,  # Procesa todos los rostros
            paste_back=True
        )

        if isinstance(output, list):
            output = output[0]

        return Image.fromarray(np.uint8(np.clip(output, 0, 255)))

    except Exception as e:
        raise RuntimeError(f"Error GFPGAN: {e}")

# -------------------- Real-ESRGAN --------------------
_realesrgan_available = False
def _load_realesrgan():
    global _realesrgan_available
    if _realesrgan_available or not torch_available:
        return _realesrgan_available
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        _realesrgan_available = True
        log_model("realesrgan", True)
        return True
    except Exception as e:
        log_model("realesrgan", False, str(e))
        return False

def upscale_imagen_realesrgan(img: Image.Image, scale: int = 4) -> Image.Image:
    if not _load_realesrgan():
        raise RuntimeError("Real-ESRGAN no disponible")
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = "RealESRGAN_x4plus.pth"
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model_path
            )

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)

        output, _ = upsampler.enhance(np.array(img), outscale=scale)
        return Image.fromarray(np.uint8(np.clip(output,0,255)))

    except Exception as e:
        raise RuntimeError(f"Error Real-ESRGAN: {e}")

# -------------------- Stable Diffusion + ControlNet --------------------
_stable_diffusion_available = False
def _load_stable_diffusion():
    global _stable_diffusion_available
    if _stable_diffusion_available or not torch_available:
        return _stable_diffusion_available
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        _stable_diffusion_available = True
        log_model("stable_diffusion", True)
        return True
    except Exception as e:
        log_model("stable_diffusion", False, str(e))
        return False

def restaurar_imagen_sd(img: Image.Image, hf_token: str,
                        prompt: str = "Restaurar imagen antigua dañada, reparar rasguños y arrugas, mejorar nitidez y colores",
                        strength: float = 0.6, steps: int = 20) -> Image.Image:
    if not hf_token:
        raise RuntimeError("HF_TOKEN no configurado")
    if not _load_stable_diffusion():
        raise RuntimeError("Stable Diffusion no disponible")
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        use_controlnet = False
        try:
            from diffusers import ControlNetModel
            use_controlnet = True
        except ImportError:
            pass

        import cv2
        import torch
        device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_controlnet:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if device_type.type=="cuda" else torch.float32,
                token=hf_token
            )
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if device_type.type=="cuda" else torch.float32,
                token=hf_token
            )
            pipe.to(device_type)
            canny_img = cv2.Canny(np.array(img), 100, 200)
            canny_img = Image.fromarray(canny_img)
            result = pipe(prompt=prompt, image=img, control_image=canny_img,
                          strength=strength, guidance_scale=7.5,
                          num_inference_steps=steps).images[0]
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device_type.type=="cuda" else torch.float32,
                token=hf_token
            )
            pipe.to(device_type)
            result = pipe(prompt=prompt, image=img, strength=strength,
                          guidance_scale=7.5, num_inference_steps=steps).images[0]

        return result

    except Exception as e:
        raise RuntimeError(f"Error Stable Diffusion: {e}")

# -------------------- Flujo completo --------------------
def restaurar_imagen_completa(img: Image.Image, hf_token: str, upscale: int = 4,
                              sd_strength: float = 0.6, sd_steps: int = 20) -> Image.Image:
    """
    Flujo completo: GFPGAN full-image → Real-ESRGAN → Stable Diffusion + ControlNet
    """
    print("🔹 GFPGAN full-image...")
    img_gfpgan = restaurar_imagen_gfpgan(img)
    print("🔹 Real-ESRGAN upscale...")
    img_upscaled = upscale_imagen_realesrgan(img_gfpgan, scale=upscale)
    print("🔹 Stable Diffusion + ControlNet...")
    img_sd = restaurar_imagen_sd(img_upscaled, hf_token, strength=sd_strength, steps=sd_steps)
    print("✅ Restauración completa finalizada")
    return img_sd
