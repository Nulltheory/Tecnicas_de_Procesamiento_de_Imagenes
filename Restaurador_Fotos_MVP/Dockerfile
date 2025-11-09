FROM python:3.9-slim

WORKDIR /app

# Instalar las dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# MÉTODO ALTERNATIVO: Instalar PyTorch desde conda-forge (más confiable)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html || \
    pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Verificar que PyTorch se instaló correctamente
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); x = torch.randn(3, 3); print('PyTorch test passed')" || \
    (echo "PyTorch installation failed" && exit 1)

# Instalar TODAS las dependencias en un solo comando para mejor resolución
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    scipy \
    matplotlib \
    requests \
    opencv-python-headless \
    streamlit \
    basicsr \
    gfpgan \
    realesrgan \
    facexlib \
    transformers \
    diffusers \
    huggingface-hub \
    google-generativeai \
    lpips \
    tqdm

# Verificar instalación completa con manejo de errores
RUN python -c "
try:
    import torch
    import numpy as np
    import cv2
    import basicsr
    import gfpgan
    print('✅ All core dependencies installed successfully')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    exit(1)
"

# Copiar la app
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
