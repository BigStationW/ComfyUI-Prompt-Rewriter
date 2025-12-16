import os
import glob

def get_matching_mmproj(model_name):
    """
    Automatically find matching mmproj file for a given model.
    Extracts the base model name and searches for corresponding mmproj file.
    
    Examples: 
        "Qwen3-VL-4B-Thinking-Q8_0.gguf" -> "Qwen3-VL-4B-Thinking-mmproj-*.gguf"
        "Qwen3-VL-8B-Thinking-UD-Q8_K_XL.gguf" -> "Qwen3-VL-8B-Thinking-mmproj-*.gguf"
    """
    if not model_name:
        return None
    
    import re
    
    # Remove .gguf extension
    base_name = model_name.replace('.gguf', '')
    
    # Remove quantization suffixes (comprehensive pattern)
    # Handles: Q8_0, Q4_K_M, Q4_K_S, Q8_K_XL, IQ4_XS, IQ3_XXS, UD-Q8_K_XL, etc.
    base_name = re.sub(r'(-UD)?-[QI]Q?\d+_\w+$', '', base_name, flags=re.IGNORECASE)
    
    # Remove float format suffixes: -F16, -BF16, -F32, etc.
    base_name = re.sub(r'-B?F\d+$', '', base_name, flags=re.IGNORECASE)
    
    # Get all mmproj files
    mmproj_files = get_mmproj_models()
    
    if not mmproj_files:
        print(f"[Model Manager] No mmproj files found in models directory")
        return None
    
    # Search for matching mmproj file - match on base model name
    # The mmproj must contain the base model name (without the mmproj part)
    matching_files = []
    for mmproj_file in mmproj_files:
        # Extract the base name of the mmproj (before "-mmproj")
        mmproj_base = re.sub(r'-mmproj.*$', '', mmproj_file, flags=re.IGNORECASE)
        
        # Check if base names match
        if base_name.lower() == mmproj_base.lower():
            matching_files.append(mmproj_file)
    
    if matching_files:
        # Take the first one found (you can add priority logic if needed)
        selected = matching_files[0]
        print(f"[Model Manager] Auto-detected mmproj: {selected} for model: {model_name}")
        if len(matching_files) > 1:
            print(f"[Model Manager] Other available mmproj variants: {matching_files[1:]}")
        return selected
    
    # Fallback: partial match search (less strict)
    for mmproj_file in mmproj_files:
        mmproj_base = re.sub(r'-mmproj.*$', '', mmproj_file, flags=re.IGNORECASE)
        if mmproj_base.lower() in base_name.lower() or base_name.lower() in mmproj_base.lower():
            print(f"[Model Manager] Detected mmproj: {mmproj_file} for model: {model_name}")
            return mmproj_file
    
    print(f"[Model Manager] No matching mmproj found for: {model_name} (base: {base_name})")
    print(f"[Model Manager] Available mmproj files: {mmproj_files}")
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
