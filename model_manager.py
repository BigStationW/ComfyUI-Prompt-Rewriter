import os
import glob

def get_matching_mmproj(model_name):
    """
    Automatically find matching mmproj file for a given model.
    Extracts the base model name and searches for corresponding mmproj file.
    
    Examples: 
        "Qwen3-VL-4B-Thinking-Q8_0.gguf" -> "Qwen3-VL-4B-Thinking-mmproj-F16.gguf"
        "Qwen3-VL-4B-Thinking-UD-Q8_K_XL.gguf" -> "Qwen3-VL-4B-Thinking-mmproj-BF16.gguf"
    """
    if not model_name:
        return None
    
    import re
    
    # Remove .gguf extension
    base_name = model_name.replace('.gguf', '')
    
    # Remove quantization suffixes (comprehensive pattern)
    # Handles: Q8_0, Q4_K_M, Q4_K_S, Q8_K_XL, IQ4_XS, IQ3_XXS, UD-Q8_K_XL, etc.
    # 
    # Pattern breakdown:
    # (-UD)?     - Optional UD (Unidiffuser) prefix  
    # -[QI]      - Dash followed by Q or I
    # Q?         - Optional second Q (for IQ patterns like IQ4_XS)
    # \d+        - One or more digits
    # _\w+       - Underscore followed by word chars (handles K_M, K_XL, _0, _XS, XXS, etc.)
    base_name = re.sub(r'(-UD)?-[QI]Q?\d+_\w+$', '', base_name, flags=re.IGNORECASE)
    
    # Remove float format suffixes: -F16, -BF16, -F32, etc.
    base_name = re.sub(r'-B?F\d+$', '', base_name, flags=re.IGNORECASE)
    
    # Get all mmproj files
    mmproj_files = get_mmproj_models()
    
    # Search for matching mmproj file
    for mmproj_file in mmproj_files:
        if base_name.lower() in mmproj_file.lower() and 'mmproj' in mmproj_file.lower():
            print(f"[Model Manager] Auto-detected mmproj: {mmproj_file} for model: {model_name}")
            return mmproj_file
    
    return None

def get_mmproj_models():
    """Get list of mmproj files in the models folder"""
    models_dir = get_models_directory()
    mmproj_files = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.gguf') and 'mmproj' in file.lower():
                mmproj_files.append(file)
    
    return sorted(mmproj_files)


def get_mmproj_path(mmproj_name):
    """Get full path to an mmproj file"""
    return os.path.join(get_models_directory(), mmproj_name)

def get_models_directory():
    """Get the path to the models directory (ComfyUI/models/LLM/gguf)"""
    # Get the custom_nodes directory (parent of this extension)
    custom_nodes_dir = os.path.dirname(os.path.dirname(__file__))
    # Go up to ComfyUI root
    comfyui_root = os.path.dirname(custom_nodes_dir)
    # Path to models/LLM/gguf
    models_dir = os.path.join(comfyui_root, "models", "LLM", "gguf")

    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir

def get_local_models():
    """Get list of local .gguf model files"""
    models_dir = get_models_directory()

    gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    # Return just the filenames, not full paths
    return [os.path.basename(f) for f in sorted(gguf_files)]

def get_huggingface_models():
    """Get list of predefined models available for download - DISABLED"""
    return []

def get_all_models():
    """Get list of local models only"""
    local_models = get_local_models()
    return local_models

def is_model_local(model_name):
    """Check if a model exists locally"""
    models_dir = get_models_directory()
    model_path = os.path.join(models_dir, model_name)
    return os.path.exists(model_path)

def get_model_path(model_name):
    """Get the full path to a model file"""
    models_dir = get_models_directory()
    return os.path.join(models_dir, model_name)
