from PIL import Image
import io

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))

def resize_image(img: Image.Image, max_size: int = 1024) -> Image.Image:
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1.0:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    return img