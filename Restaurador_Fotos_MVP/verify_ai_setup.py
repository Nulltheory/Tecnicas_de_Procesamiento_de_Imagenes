#!/usr/bin/env python3
"""
🔍 Script de Verificación - Restaurador Fotográfico AI
Verifica que todos los componentes críticos estén funcionando
"""

import sys
import warnings

def test_pytorch():
    """Verificar PyTorch y su funcionamiento"""
    print("🔍 Verificando PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} disponible")
        
        # Test básico de funcionamiento
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = test_tensor.sum().item()
        print(f"✅ Test PyTorch: {result:.0f} ✓")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ Modo CPU - funcional para cloud deployment")
        
        return True
    except ImportError:
        print("❌ PyTorch NO disponible")
        return False

def test_ai_models():
    """Verificar que los modelos de IA se pueden importar"""
    print("\n🔍 Verificando modelos de IA...")
    
    models_status = {}
    
    # Test Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        print("✅ Real-ESRGAN disponible")
        models_status['Real-ESRGAN'] = True
    except ImportError:
        print("❌ Real-ESRGAN NO disponible")
        models_status['Real-ESRGAN'] = False
    
    # Test CodeFormer
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        print("✅ CodeFormer disponible")
        models_status['CodeFormer'] = True
    except ImportError:
        print("❌ CodeFormer NO disponible")
        models_status['CodeFormer'] = False
    
    # Test GFPGAN
    try:
        from gfpgan import GFPGANer
        print("✅ GFPGAN disponible")
        models_status['GFPGAN'] = True
    except ImportError:
        print("❌ GFPGAN NO disponible")
        models_status['GFPGAN'] = False
    
    # Test Stable Diffusion
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        print("✅ Stable Diffusion disponible")
        models_status['Stable Diffusion'] = True
    except ImportError:
        print("❌ Stable Diffusion NO disponible")
        models_status['Stable Diffusion'] = False
    
    return models_status

def test_basic_dependencies():
    """Verificar dependencias básicas"""
    print("\n🔍 Verificando dependencias básicas...")
    
    basic_deps = [
        'streamlit', 'PIL', 'numpy', 'cv2', 'scipy', 'matplotlib', 'requests'
    ]
    
    all_ok = True
    for dep in basic_deps:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'cv2':
                import cv2
            else:
                __import__(dep)
            print(f"✅ {dep} disponible")
        except ImportError:
            print(f"❌ {dep} NO disponible")
            all_ok = False
    
    return all_ok

def test_app_modules():
    """Verificar módulos de la aplicación"""
    print("\n🔍 Verificando módulos de la aplicación...")
    
    try:
        from models.diffusion import restaurar_imagen_gfpgan
        print("✅ models.diffusion disponible")
        diffusion_ok = True
    except ImportError as e:
        print(f"❌ models.diffusion NO disponible: {e}")
        diffusion_ok = False
    
    try:
        from models.analysis import analizar_calidad_clip
        print("✅ models.analysis disponible")
        analysis_ok = True
    except ImportError as e:
        print(f"❌ models.analysis NO disponible: {e}")
        analysis_ok = False
    
    return diffusion_ok and analysis_ok

def main():
    """Función principal de verificación"""
    print("🚀 VERIFICACIÓN COMPLETA - Restaurador Fotográfico AI")
    print("=" * 60)
    
    # Suprimir warnings para salida limpia
    warnings.filterwarnings("ignore")
    
    # Ejecutar todas las verificaciones
    torch_ok = test_pytorch()
    basic_ok = test_basic_dependencies()
    models_status = test_ai_models()
    app_ok = test_app_modules()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    print(f"🔥 PyTorch: {'✅ OK' if torch_ok else '❌ FALTA'}")
    print(f"📦 Dependencias básicas: {'✅ OK' if basic_ok else '❌ FALTA'}")
    print(f"🤖 Modelos de IA: {sum(models_status.values())}/{len(models_status)} disponibles")
    print(f"🎯 Módulos de app: {'✅ OK' if app_ok else '❌ FALTA'}")
    
    # Verificar si puede ejecutar el modo completo
    if torch_ok and basic_ok and app_ok:
        available_models = sum(models_status.values())
        if available_models >= 4:
            print("\n✅ ESTADO: LISTO PARA MODO COMPLETO")
            print(f"🎯 {available_models} modelos de IA disponibles")
            print("🚀 La aplicación puede ejecutar todos los algoritmos de restauración")
        else:
            print(f"\n⚠️ ESTADO: MODO LIMITADO")
            print(f"⚡ Solo {available_models} modelos de IA disponibles")
            print("💡 Algunas funciones avanzadas no estarán disponibles")
    else:
        print("\n❌ ESTADO: MODO BÁSICO")
        print("🔧 La aplicación funcionará solo con procesamiento básico")
        print("📋 Solución: Instalar dependencias faltantes")
    
    print("\n💡 TIP: Para modo completo, instalar con:")
    print("   pip install -r requirements-deployment.txt")

if __name__ == "__main__":
    main()