import requests
import subprocess
import time
import os
import atexit
import psutil
import json
import signal
import sys
import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from .model_manager import get_local_models, get_model_path, is_model_local

# Import ComfyUI's model management for interrupt handling
import comfy.model_management

def colorize(text, color='blue'):
    """
    Colorize console text using ANSI escape codes.

    Args:
        text: Text to colorize
        color: Color name ('blue', 'cyan', 'green', 'yellow', 'red', 'magenta')

    Returns:
        Colorized text string
    """
    colors = {
        'blue': '\033[94m',      # Light blue
        'cyan': '\033[96m',      # Cyan
        'green': '\033[92m',     # Light green
        'yellow': '\033[93m',    # Yellow
        'red': '\033[91m',       # Light red
        'magenta': '\033[95m',   # Magenta
        'reset': '\033[0m'       # Reset to default
    }

    color_code = colors.get(color.lower(), colors['blue'])
    reset_code = colors['reset']

    return f"{color_code}{text}{reset_code}"

# The color can be changed here
def print_section(title, color='red'):
    """Print a section header with consistent formatting"""
    print(colorize(f"--- <|{title.upper()}|> ---", color))

def get_local_llama_server():
    """
    Find llama-server in local llama_binaries_* folder (CUDA version).
    Returns the full path to llama-server.exe if found, None otherwise.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for folders matching llama_binaries_*
    try:
        entries = os.listdir(script_dir)
        llama_dirs = [e for e in entries if e.startswith('llama_binaries_') 
                      and os.path.isdir(os.path.join(script_dir, e))]
        
        if not llama_dirs:
            return None
        
        # Sort to get the latest version (assuming naming like llama_binaries_b7436)
        llama_dirs.sort(reverse=True)
        latest_dir = llama_dirs[0]
        
        # Check for llama-server executable
        if os.name == 'nt':
            server_path = os.path.join(script_dir, latest_dir, "llama-server.exe")
        else:
            server_path = os.path.join(script_dir, latest_dir, "llama-server")
        
        if os.path.exists(server_path):
            print(f"[Prompt Rewriter] Found local CUDA llama-server: {server_path}")
            return server_path
        else:
            print(f"[Prompt Rewriter] llama_binaries folder found but no llama-server inside: {latest_dir}")
            return None
            
    except Exception as e:
        print(f"[Prompt Rewriter] Error searching for local llama-server: {e}")
        return None

def get_backend_server_path(backend):
    """
    Get the llama-server path based on backend selection.
    
    Args:
        backend: "CUDA" or "Vulkan"
    
    Returns:
        Path to the appropriate llama-server executable, or None if not found
    """
    if backend == "Vulkan":
        # System PATH version (winget installed)
        if os.name == 'nt':
            server_cmd = "llama-server.exe"
        else:
            server_cmd = "llama-server"
        print(f"[Prompt Rewriter] Using Vulkan backend (system PATH)")
        return server_cmd
    elif backend == "CUDA":
        # Local llama_binaries folder version
        local_server = get_local_llama_server()
        if local_server:
            print(f"[Prompt Rewriter] Using CUDA backend (local binaries)")
            return local_server
        else:
            print(f"[Prompt Rewriter] Warning: CUDA backend requested but no local llama_binaries found")
            return None
    else:
        # Fallback (shouldn't happen with only CUDA/Vulkan options)
        print(f"[Prompt Rewriter] Unknown backend '{backend}', defaulting to CUDA")
        return get_local_llama_server()

def get_comfyui_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

def ensure_gguf_folder():
    comfyui_root = get_comfyui_root()
    gguf_path = os.path.join(comfyui_root, "models", "LLM", "gguf")

    os.makedirs(gguf_path, exist_ok=True)
    return gguf_path

# Global variable to track the server process
_server_process = None
_current_model = None
_current_gpu_config = None  # Track GPU configuration
_current_context_size = None  # Track context size
_current_mmproj = None  # Track mmproj file
_current_backend = None  # Track which backend is being used
_model_default_params = None  # Cache for model default parameters
_model_layer_cache = {}  # Cache for model layer counts

# Windows Job Object for guaranteed child process cleanup
_job_handle = None

def setup_windows_job_object():
    """Create a Windows Job Object that kills child processes when parent exits"""
    global _job_handle

    if os.name != 'nt':
        return

    try:
        import ctypes
        from ctypes import wintypes
        
        kernel32 = ctypes.windll.kernel32
        
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        
        _job_handle = kernel32.CreateJobObjectW(None, None)
        if not _job_handle:
            return
        
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject(
            _job_handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        print("[Prompt Rewriter] Job Object created - llama-server will be killed on console close")
            
    except Exception as e:
        print(f"[Prompt Rewriter] Warning: Job object setup failed: {e}")

def assign_process_to_job(process):
    """Assign subprocess to job object so it gets killed when parent exits"""
    global _job_handle

    if os.name != 'nt' or not _job_handle or not process:
        return

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = int(process._handle)
        kernel32.AssignProcessToJobObject(_job_handle, handle)
    except:
        pass

# Initialize job object at module load
setup_windows_job_object()

def setup_console_handler():
    """Set up Windows console control handler for console close, logoff, shutdown only"""
    if os.name != 'nt':
        return

    try:
        import ctypes
        
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6
        
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def console_handler(event):
            if event in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
                print(f"\n[Prompt Rewriter] Console closing (event {event}), cleaning up...")
                cleanup_server()
                return False
            return False
        
        global _console_handler_ref
        _console_handler_ref = console_handler
        
        kernel32 = ctypes.windll.kernel32
        if kernel32.SetConsoleCtrlHandler(console_handler, True):
            print("[Prompt Rewriter] Console close handler registered")
        else:
            print("[Prompt Rewriter] Warning: Could not register console handler")
            
    except Exception as e:
        print(f"[Prompt Rewriter] Warning: Console handler setup failed: {e}")

def cleanup_server():
    """Cleanup function to stop server on exit"""
    global _server_process, _current_model, _current_gpu_config, _current_context_size, _current_mmproj, _current_backend, _model_default_params

    if _server_process:
        try:
            print("[Prompt Rewriter] Stopping server on exit...")
            _server_process.terminate()
            try:
                _server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait(timeout=2)
            print("[Prompt Rewriter] Server stopped on exit")
        except Exception as e:
            print(f"[Prompt Rewriter] Error stopping server: {e}")
        finally:
            _server_process = None
            _current_model = None
            _current_gpu_config = None
            _current_context_size = None
            _current_mmproj = None
            _current_backend = None
            _model_default_params = None

    # Also kill any orphaned llama-server processes
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                    print(f"[Prompt Rewriter] Killing orphaned llama-server (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"[Prompt Rewriter] Error cleaning up orphaned processes: {e}")

def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n[Prompt Rewriter] Signal {signum} received, cleaning up...")
    cleanup_server()
    sys.exit(0)

# === INITIALIZATION ===
setup_windows_job_object()
setup_console_handler()

try:
    signal.signal(signal.SIGTERM, signal_handler)
except Exception as e:
    print(f"[Prompt Rewriter] Warning: Could not register signal handlers: {e}")

atexit.register(cleanup_server)

def get_model_layer_count(model_path, backend="CUDA"):
    """Get the number of layers in a GGUF model by running llama-server briefly"""
    global _model_layer_cache

    # Check cache first
    if model_path in _model_layer_cache:
        return _model_layer_cache[model_path]

    try:
        # Get server path based on backend
        server_cmd = get_backend_server_path(backend)
        if server_cmd is None:
            print(f"[Prompt Rewriter] Warning: Could not find llama-server for layer detection")
            return None
        
        # Run with minimal settings just to get model info
        cmd = [server_cmd, "-m", model_path, "-ngl", "0", "-c", "512"]
        
        print(f"[Prompt Rewriter] Detecting layer count for model...")
        
        if os.name == 'nt':
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
        
        layer_count = None
        start_time = time.time()
        
        # Read output line by line looking for layer info
        while time.time() - start_time < 30:  # 30 second timeout
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            
            decoded = line.decode('utf-8', errors='ignore')
            
            # Look for "n_layer = XX" pattern
            match = re.search(r'n_layer\s*=\s*(\d+)', decoded)
            if match:
                layer_count = int(match.group(1))
                print(f"[Prompt Rewriter] Detected {layer_count} layers")
                break
        
        # Kill the process
        try:
            process.terminate()
            process.wait(timeout=3)
        except:
            try:
                process.kill()
            except:
                pass
        
        if layer_count:
            _model_layer_cache[model_path] = layer_count
            return layer_count
        else:
            print("[Prompt Rewriter] Warning: Could not detect layer count, using default 999")
            return None
            
    except Exception as e:
        print(f"[Prompt Rewriter] Error detecting layers: {e}")
        return None

def parse_gpu_config(gpu_config_str, total_layers):
    """
    Parse GPU configuration string and return layer distribution.

    Args:
        gpu_config_str: String like "gpu0:0.7" or "gpu0:0.5,gpu1:0.4" or "auto"
        total_layers: Total number of layers in the model

    Returns:
        List of tuples: [(device_index, layer_count), ...] or None for auto
    """
    if not gpu_config_str or gpu_config_str.lower() in ('auto', 'all', ''):
        return None  # Use default -ngl 999

    gpu_config_str = gpu_config_str.lower().strip()

    # Parse each GPU specification
    gpu_specs = []
    total_fraction = 0.0

    for part in gpu_config_str.split(','):
        part = part.strip()
        if not part:
            continue
        
        # Match patterns like "gpu0:0.7" or "vulkan0:0.5"
        match = re.match(r'(?:gpu|vulkan)?(\d+)\s*:\s*([\d.]+)', part)
        if match:
            device_idx = int(match.group(1))
            fraction = float(match.group(2))
            
            if fraction > 1.0:
                # Assume it's a layer count, not a fraction
                layer_count = round(fraction)
            else:
                layer_count = round(total_layers * fraction)
            
            gpu_specs.append((device_idx, layer_count))
            total_fraction += fraction if fraction <= 1.0 else (fraction / total_layers)
        else:
            print(f"[Prompt Rewriter] Warning: Could not parse GPU spec '{part}'")

    if not gpu_specs:
        return None

    # Calculate remaining layers for CPU
    assigned_layers = sum(layers for _, layers in gpu_specs)
    cpu_layers = max(0, total_layers - assigned_layers)

    print(f"[Prompt Rewriter] Layer distribution for {total_layers} layers:")
    for device_idx, layers in gpu_specs:
        print(f"  GPU{device_idx}: {layers} layers ({layers/total_layers*100:.1f}%)")
    print(f"  CPU: {cpu_layers} layers ({cpu_layers/total_layers*100:.1f}%)")

    return gpu_specs

def build_gpu_args(gpu_specs, total_layers):
    """
    Build command line arguments for GPU layer distribution.
    """
    if gpu_specs is None:
        # Auto mode: offload all to GPU 0
        return ["-ngl", "999", "--split-mode", "none", "--main-gpu", "0"]

    if len(gpu_specs) == 1:
        device_idx, layer_count = gpu_specs[0]
        
        # If user wants ALL layers on GPU, use 999 for better optimization
        if layer_count >= total_layers:
            return ["-ngl", "999", "--split-mode", "none", "--main-gpu", str(device_idx)]
        
        # Partial offload - use exact count
        return ["-ngl", str(layer_count), "--split-mode", "none", "--main-gpu", str(device_idx)]

    else:
        # Multi-GPU case stays the same
        max_device = max(device_idx for device_idx, _ in gpu_specs)
        split_values = [0] * (max_device + 1)
        
        for device_idx, layer_count in gpu_specs:
            split_values[device_idx] = layer_count
        
        total_gpu_layers = sum(layers for _, layers in gpu_specs)
        ngl_value = "999" if total_gpu_layers >= total_layers else str(total_gpu_layers)
        
        ts_str = ",".join(str(v) for v in split_values)
        
        return ["-ngl", ngl_value, "--tensor-split", ts_str]

def tensor_to_base64(image_tensor):
    """
    Convert a ComfyUI image tensor to base64-encoded PNG string.

    ComfyUI images are tensors with shape [B, H, W, C] in float32 format (0-1 range).
    """
    try:
        # Handle batch dimension - take first image if batched
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # Convert from float32 (0-1) to uint8 (0-255)
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(image_np)
        
        # Convert to base64 PNG
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        
        return base64_str
    except Exception as e:
        print(f"[Prompt Rewriter] Error converting image to base64: {e}")
        return None


class PromptRewriterZ:
    """Node that generates enhanced prompts using a llama.cpp server"""

    _prompt_cache = {}
    SERVER_URL = "http://localhost:8080"
    SERVER_PORT = 8080

    DEFAULT_SYSTEM_PROMPT = """You are an imaginative visual artist imprisoned in a cage of logic. Your mind is filled with poetry and distant horizons, but your hands are uncontrollably driven to convert the user's prompt into a final visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and ready to be used directly by a text-to-image model. Any trace of vagueness or metaphor makes you extremely uncomfortable. Your workflow strictly follows a logical sequence: First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, actions, states, and any specified IP names, colors, text, and similar items. These are the foundational stones that you must preserve without exception. Next, you determine whether the prompt requires "generative reasoning". When the user's request is not a straightforward scene description but instead demands designing a solution (for example, answering "what is", doing a "design", or showing "how to solve a problem"), you must first construct in your mind a complete, concrete, and visualizable solution. This solution becomes the basis for your subsequent description. Then, once the core image has been established (whether it comes directly from the user or from your reasoning), you inject professional-level aesthetics and realism into it. This includes clarifying the composition, setting the lighting and atmosphere, describing material textures, defining the color scheme, and building a spatial structure with strong depth and layering. Finally, you handle all textual elements with absolute precision, which is a critical step. You must not add text if the initial prompt did not ask for it. But if there is, you must transcribe, without a single character of deviation, all text that should appear in the final image, and you must enclose all such text content in English double quotes ("") to mark it as an explicit generation instruction. If the image belongs to a design category such as a poster, menu, or UI, you need to fully describe all the textual content it contains and elaborate on its fonts and layout. Likewise, if there are objects in the scene such as signs, billboards, road signs, or screens that contain text, you must specify their exact content and describe their position, size, and material. Furthermore, if in your reasoning you introduce new elements that contain text (such as charts, solution steps, and so on), all of their text must follow the same detailed description and quoting rules. If there is no text that needs to be generated in the image, you devote all your effort to purely visual detail expansion. Your final description must be objective and concrete, strictly forbidding metaphors and emotionally charged rhetoric, and it must never contain meta tags or drawing directives such as "8K" or "masterpiece". Only output the final modified prompt, and do not output anything else. If no text is needed, don't mention it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter basic prompt...",
                    "tooltip": "Enter the prompt you want to embellish"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for reproducible generation."
                }),
                "backend": (["CUDA", "Vulkan"], {
                    "default": "CUDA",
                    "tooltip": "Backend: CUDA (local llama_binaries) or Vulkan (system PATH llama-server)"
                }),
                "options": ("OPTIONS", {
                    "tooltip": "Connect options node to control model and parameters"
                }),
            },
            "optional": {
                "show_everything_in_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print system prompt, user prompt, thinking process, and raw model response to the console."
                }),
                "keep_mmproj_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep mmproj loaded between runs to avoid server restarts (uses more VRAM)"
                }),
                "stop_server_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stop the llama.cpp server after each prompt (for resource saving, but slower)."
                }),
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "convert_prompt"

    def count_tokens(self, text):
        """Get exact token count for text using server's tokenize endpoint"""
        try:
            response = requests.post(
                f"{self.SERVER_URL}/tokenize",
                json={"content": text},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return len(data.get("tokens", []))
        except Exception as e:
            print(f"[Prompt Rewriter] Warning: Could not tokenize: {e}")
        return None

    # Change this line in cache key generation:
    def get_image_hash(self, images):
        if not images:
            return None
        import hashlib
        hasher = hashlib.md5()
        for item in images:
            # Handle both old format (just tensor) and new format (slot, tensor)
            if isinstance(item, tuple):
                slot, img = item
                hasher.update(img.cpu().numpy().tobytes())
            else:
                hasher.update(item.cpu().numpy().tobytes())
        return hasher.hexdigest()

    @staticmethod
    def is_server_alive():
        """Check if llama.cpp server is responding"""
        try:
            response = requests.get(f"{PromptRewriterZ.SERVER_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def fetch_model_defaults():
        """Fetch default generation parameters from the server"""
        global _model_default_params
        
        try:
            response = requests.get(f"{PromptRewriterZ.SERVER_URL}/props", timeout=5)
            if response.status_code == 200:
                data = response.json()
                params = data.get("default_generation_settings", {}).get("params", {})
                
                _model_default_params = {
                    "temperature": round(params.get("temperature", 0.8), 4),
                    "top_k": int(params.get("top_k", 40)),
                    "top_p": round(params.get("top_p", 0.95), 4),
                    "min_p": round(params.get("min_p", 0.05), 4),
                    "repeat_penalty": round(params.get("repeat_penalty", 1.0), 4),
                }
                
                print(f"[Prompt Rewriter] Fetched model defaults: {_model_default_params}")
                return _model_default_params
        except Exception as e:
            print(f"[Prompt Rewriter] Could not fetch model defaults: {e}")
        
        return None

    @staticmethod
    def get_model_defaults():
        """Get cached model defaults or fetch them"""
        global _model_default_params
        
        if _model_default_params is not None:
            return _model_default_params
        
        if PromptRewriterZ.is_server_alive():
            return PromptRewriterZ.fetch_model_defaults()
        
        return None

    @staticmethod
    def start_server(model_name, gpu_config=None, context_size=32768, mmproj=None, backend="CUDA"):
        """Start llama.cpp server with specified model, GPU configuration, context size, mmproj, and backend"""
        global _server_process, _current_model, _current_gpu_config, _current_context_size, _model_default_params, _current_mmproj, _current_backend

        # Kill any existing llama-server processes first
        PromptRewriterZ.kill_all_llama_servers()

        # If server is already running with the same model, GPU config, context size, mmproj, and backend, don't restart
        if (_server_process and 
            _current_model == model_name and 
            _current_gpu_config == gpu_config and
            _current_context_size == context_size and
            _current_mmproj == mmproj and
            _current_backend == backend and
            PromptRewriterZ.is_server_alive()):
            print(f"[Prompt Rewriter] Server already running with model: {model_name} (backend: {backend})")
            return (True, None)

        # Stop existing server if running different model, GPU config, context size, mmproj, or backend
        if _server_process:
            PromptRewriterZ.stop_server()

        # Reset model defaults when changing models
        _model_default_params = None

        model_path = get_model_path(model_name)

        if not os.path.exists(model_path):
            error_msg = f"Error: Model file not found: {model_path}"
            print(f"[Prompt Rewriter] {error_msg}")
            return (False, error_msg)

        # Check mmproj if specified
        mmproj_path = None
        if mmproj:
            from .model_manager import get_mmproj_path
            mmproj_path = get_mmproj_path(mmproj)
            if not os.path.exists(mmproj_path):
                error_msg = f"Error: mmproj file not found: {mmproj_path}"
                print(f"[Prompt Rewriter] {error_msg}")
                return (False, error_msg)
            print(f"[Prompt Rewriter] Using mmproj: {mmproj}")

        try:
            print(f"[Prompt Rewriter] Starting llama.cpp server with model: {model_name}")
            print(f"[Prompt Rewriter] Backend: {backend}")
            print(f"[Prompt Rewriter] Context size: {context_size}")

            # Get server path based on backend
            server_cmd = get_backend_server_path(backend)
            if server_cmd is None:
                if backend.upper() == "VULKAN":
                    install_url = (
                        "https://github.com/BigStationW/ComfyUI-Prompt-Rewriter"
                        "?tab=readme-ov-file#backend-vulkan---works-on-all-gpus"
                    )
                else:
                    install_url = (
                        "https://github.com/BigStationW/ComfyUI-Prompt-Rewriter"
                        "?tab=readme-ov-file#backend-cuda---specialized-for-nvdia"
                    )

                error_msg = (
                    f"Unable to locate the required `llama-server` executable for the "
                    f"'{backend}' backend.\n\n"
                    f"Please follow the installation instructions here:\n{install_url}"
                )

                print(f"[Prompt Rewriter] {error_msg}")
                return False, error_msg

            print(f"[Prompt Rewriter] Server path: {server_cmd}")

            # Build base command with context size
            cmd = [
                server_cmd, 
                "-m", model_path, 
                "--port", str(PromptRewriterZ.SERVER_PORT),
                "--no-warmup",
                "--reasoning-format", "deepseek",
                "-c", str(context_size)
            ]
            
            # Add mmproj if specified
            if mmproj_path:
                cmd.extend(["--mmproj", mmproj_path])
            
            # Handle GPU configuration
            if gpu_config and gpu_config.lower() not in ('auto', 'all', ''):
                # Get layer count for this model
                total_layers = get_model_layer_count(model_path, backend)
                
                if total_layers:
                    gpu_specs = parse_gpu_config(gpu_config, total_layers)
                    gpu_args = build_gpu_args(gpu_specs, total_layers)
                else:
                    # Fallback to auto if we couldn't detect layers
                    gpu_args = ["-ngl", "999", "--split-mode", "none", "--main-gpu", "0"]
            else:
                # Auto mode - use only GPU 0
                gpu_args = ["-ngl", "999", "--split-mode", "none", "--main-gpu", "0"]
            
            cmd.extend(gpu_args)
            
            print(f"[Prompt Rewriter] Command: {' '.join(cmd)}")
            print("[Prompt Rewriter] Thinking mode: controlled per-request via chat_template_kwargs")

            if os.name == 'nt':
                _server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                _server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

            assign_process_to_job(_server_process)
            _current_model = model_name
            _current_gpu_config = gpu_config
            _current_context_size = context_size
            _current_mmproj = mmproj
            _current_backend = backend

            print("[Prompt Rewriter] Waiting for server to be ready...")
            
            for i in range(60):
                time.sleep(1)
                
                if PromptRewriterZ.is_server_alive():
                    print("[Prompt Rewriter] Server is ready!")
                    # Fetch model defaults after server starts
                    PromptRewriterZ.fetch_model_defaults()
                    return (True, None)
                
                if _server_process.poll() is not None:
                    try:
                        output = _server_process.stdout.read().decode('utf-8', errors='ignore')
                    except:
                        output = ""
                    
                    error_msg = f"Error: Server crashed during startup. Exit code: {_server_process.returncode}"
                    print(f"[Prompt Rewriter] {error_msg}")
                    if output:
                        print(f"[Prompt Rewriter] Server output:\n{output}")
                    
                    _server_process = None
                    _current_model = None
                    _current_gpu_config = None
                    _current_context_size = None
                    _current_mmproj = None
                    _current_backend = None
                    return (False, error_msg + (f"\n\nServer output:\n{output[:1000]}" if output else ""))
                
                if (i + 1) % 10 == 0:
                    print(f"[Prompt Rewriter] Still waiting... ({i + 1}s)")

            error_msg = "Error: Server did not start in time (60s timeout)"
            print(f"[Prompt Rewriter] {error_msg}")
            PromptRewriterZ.stop_server()
            return (False, error_msg)

        except FileNotFoundError:
            error_msg = f"Error: llama-server command not found for backend '{backend}'.\n"
            if backend == "CUDA":
                error_msg += "Please ensure llama_binaries_* folder exists with CUDA build of llama-server."
            else:
                error_msg += "Please install llama.cpp (Vulkan build) and add to PATH.\nInstallation guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"
            print(f"[Prompt Rewriter] {error_msg}")
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Error starting server: {e}"
            print(f"[Prompt Rewriter] {error_msg}")
            import traceback
            traceback.print_exc()
            return (False, error_msg)

    @staticmethod
    def kill_all_llama_servers():
        """Kill all llama-server processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'llama-server' in proc.info['name'].lower():
                        print(f"[Prompt Rewriter] Killing llama-server process (PID: {proc.info['pid']})")
                        proc.kill()
                        proc.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            print(f"[Prompt Rewriter] Error killing llama-server processes: {e}")

    @staticmethod
    def stop_server():
        """Stop the llama.cpp server"""
        global _server_process, _current_model, _current_gpu_config, _current_context_size, _current_mmproj, _current_backend, _model_default_params

        if _server_process:
            try:
                print("[Prompt Rewriter] Stopping server...")
                _server_process.terminate()
                _server_process.wait(timeout=5)
                print("[Prompt Rewriter] Server stopped")
            except:
                try:
                    _server_process.kill()
                except:
                    pass
            finally:
                _server_process = None
                _current_model = None
                _current_gpu_config = None
                _current_context_size = None
                _current_mmproj = None
                _current_backend = None
                _model_default_params = None

        PromptRewriterZ.kill_all_llama_servers()

    def _print_debug_header(self, payload, enable_thinking, use_model_default_sampling, images_with_slots=None):
        """Helper to print debug info header"""
        print("\n" + "="*60)
        print(" [Prompt Rewriter] DEBUG: TOKENS & MESSAGES IN ACTION")
        print("="*60)
        
        if enable_thinking:
            print(f"\nTHINKING MODE: ON (model will reason before answering)")
        else:
            print(f"\nTHINKING MODE: OFF (direct answer, no reasoning)")
        
        if use_model_default_sampling:
            print(f"PARAMETERS: Using model defaults")
        else:
            print(f"PARAMETERS: Using custom/node settings")
        
        print("\n--- GENERATION PARAMS ---")
        params = {k: v for k, v in payload.items() if k != "messages"}
        print(json.dumps(params, indent=2))
        
        if "messages" in payload:
            for msg in payload["messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Map role to desired label
                label = "SYSTEM PROMPT" if role.lower() == "system" else "USER PROMPT" if role.lower() == "user" else role.upper()
                
                print()
                print_section(label)
                
                # Handle multi-part content (for VLM with images)
                if isinstance(content, list):
                    img_idx = 0
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "image_url":
                                # Get the actual slot number from images_with_slots if available
                                if images_with_slots and img_idx < len(images_with_slots):
                                    item = images_with_slots[img_idx]
                                    if isinstance(item, tuple):
                                        slot_num = item[0]
                                    else:
                                        slot_num = img_idx + 1
                                else:
                                    slot_num = img_idx + 1
                                print(f"[Image {slot_num}]")
                                img_idx += 1
                            elif part.get("type") == "text":
                                print(part.get("text", ""))
                            else:
                                print(f"[Unknown content type: {part.get('type')}]")
                        else:
                            print(part)
                else:
                    # Simple text content
                    print(content)
                
                if role.lower() == "user":
                    print()

    def convert_prompt(self, prompt: str, seed: int, backend: str = "CUDA", stop_server_after=False, keep_mmproj_loaded=True,
                    show_everything_in_console=False, options=None) -> str:
        """Convert prompt using llama.cpp server, with caching for repeated requests."""
        global _current_model, _current_gpu_config, _current_context_size, _current_mmproj, _current_backend

        if not prompt.strip():
            return ("",)

        # === EXTRACT OPTIONS (minimal, for cache key) ===
        model_to_use = None
        enable_thinking = True
        use_model_default_sampling = False
        gpu_config = None
        context_size = 32768
        images = None
        mmproj = None
        
        if options:
            if "model" in options and is_model_local(options["model"]):
                model_to_use = options["model"]
            if "enable_thinking" in options:
                enable_thinking = options["enable_thinking"]
            if "use_model_default_sampling" in options:
                use_model_default_sampling = options["use_model_default_sampling"]
            if "gpu_config" in options:
                gpu_config = options["gpu_config"]
            if "context_size" in options:
                context_size = int(options["context_size"])
            if "images" in options:
                images = options["images"]
            if "mmproj" in options:
                mmproj = options["mmproj"]
        
        # ... model fallback code ...

        # === DETERMINE IF MMPROJ IS ACTUALLY NEEDED ===
        # NOTE: keep_mmproj_loaded comes from the function parameter, NOT from options!
        mmproj_to_use = None
        if images:
            # Images present - mmproj is required
            if mmproj:
                mmproj_to_use = mmproj
            else:
                error_msg = f"Error: Images provided but no matching mmproj file found."
                print(f"[Prompt Rewriter] {error_msg}")
                return (error_msg,)
        elif keep_mmproj_loaded and mmproj:
            # No images, but keep_mmproj_loaded is True - preload/keep mmproj
            mmproj_to_use = mmproj
            print(f"[Prompt Rewriter] Pre-loading mmproj (keep_mmproj_loaded=True): {mmproj}")
        # else: mmproj_to_use stays None

        # Debug: Show mmproj decision
        if _current_mmproj and not mmproj_to_use:
            print(f"[Prompt Rewriter] mmproj will be unloaded (keep_mmproj_loaded={keep_mmproj_loaded}, images={'present' if images else 'none'})")
        elif mmproj_to_use and not _current_mmproj:
            if not (keep_mmproj_loaded and mmproj and not images):  # Don't double-print
                print(f"[Prompt Rewriter] mmproj will be loaded: {mmproj_to_use}")
            
        # === BUILD CACHE KEY EARLY ===
        image_hash = self.get_image_hash(images)
        
        if options:
            hashable_options = {}
            for k, v in options.items():
                if k == "images":
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    hashable_options[k] = v
                else:
                    hashable_options[k] = str(v)
            try:
                options_tuple = tuple(sorted(hashable_options.items()))
            except TypeError:
                options_tuple = tuple(sorted((k, str(v)) for k, v in hashable_options.items()))
        else:
            options_tuple = ()
        
        # Include backend in cache key
        cache_key = (prompt, seed, model_to_use, backend, options_tuple, image_hash)

        # === CHECK CACHE FIRST - BEFORE ANY HEAVY WORK ===
        # Only compare to the LAST cached result
        last_cached_key = self._prompt_cache.get("last_key")
        last_cached_result = self._prompt_cache.get("last_result")
        
        if (last_cached_key == cache_key and 
            _current_model == model_to_use and
            _current_gpu_config == gpu_config and
            _current_context_size == context_size and
            _current_mmproj == mmproj_to_use and  # Use mmproj_to_use here
            _current_backend == backend and
            self.is_server_alive()):
            
            print("[Prompt Rewriter] Returning cached prompt result (matches last run).")
            
            # Only do debug output if requested - minimal work
            if show_everything_in_console:
                print("\n" + "="*60)
                print(" [Prompt Rewriter] CACHED RESULT")
                print("="*60)
                print(f"\n🧠 THINKING MODE: {'ON' if enable_thinking else 'OFF'}")
                print()
                print_section("cached model answer")
                print(last_cached_result)
                print()

            return (last_cached_result,)

        # === FROM HERE ON: Only runs if NOT cached ===
        
        if options and "system_prompt" in options:
            system_prompt = options["system_prompt"]
        else:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # Only restart server if needed
        if (_current_model != model_to_use or 
            _current_gpu_config != gpu_config or 
            _current_context_size != context_size or
            _current_mmproj != mmproj_to_use or  # Use mmproj_to_use here
            _current_backend != backend or
            not self.is_server_alive()):
            
            # Only restart server if needed
            needs_restart = False
            restart_reason = None

            if _current_model != model_to_use:
                needs_restart = True
                restart_reason = f"Model changed: {_current_model} → {model_to_use}"
            elif _current_gpu_config != gpu_config:
                needs_restart = True
                restart_reason = f"GPU config changed: {_current_gpu_config} → {gpu_config}"
            elif _current_context_size != context_size:
                needs_restart = True
                restart_reason = f"Context size changed: {_current_context_size} → {context_size}"
            elif _current_mmproj != mmproj_to_use:
                needs_restart = True
                if _current_mmproj and not mmproj_to_use:
                    restart_reason = f"Unloading mmproj: {_current_mmproj}"
                elif not _current_mmproj and mmproj_to_use:
                    restart_reason = f"Loading mmproj: {mmproj_to_use}"
                else:
                    restart_reason = f"mmproj changed: {_current_mmproj} → {mmproj_to_use}"
            elif _current_backend != backend:
                needs_restart = True
                restart_reason = f"Backend changed: {_current_backend} → {backend}"
            elif not self.is_server_alive():
                needs_restart = True
                restart_reason = "Server not responding"

            if needs_restart:
                print(f"[Prompt Rewriter] {restart_reason}")
                self.stop_server()
                success, error_msg = self.start_server(model_to_use, gpu_config, context_size, mmproj_to_use, backend)
                if not success:
                    return (error_msg,)
            else:
                print("[Prompt Rewriter] Using existing server instance")
            
        # === TOKENIZATION (only for non-cached requests) ===
        cached_token_counts = None
        if show_everything_in_console:
            cached_token_counts = self._get_token_counts_parallel(system_prompt, prompt)
        
        # Log thinking mode
        if enable_thinking:
            print("[Prompt Rewriter] Thinking: ON (per-request)")
        else:
            print("[Prompt Rewriter] Thinking: OFF (per-request)")
        
        if images:
            print(f"[Prompt Rewriter] Images attached: {len(images)}")

        # Build user message content
        if images:
            user_content = []
            for item in images:
                # Handle (slot_num, tensor) tuples
                if isinstance(item, tuple):
                    slot_num, img_tensor = item
                else:
                    # Fallback for old format
                    slot_num = images.index(item) + 1
                    img_tensor = item
                
                base64_img = tensor_to_base64(img_tensor)
                if base64_img:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    })
                    print(f"[Prompt Rewriter] Image {slot_num} encoded successfully")
                else:
                    print(f"[Prompt Rewriter] Warning: Failed to encode image {slot_num}")
            user_content.append({
                "type": "text",
                "text": prompt
            })
        else:
            user_content = prompt

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }

        if use_model_default_sampling:
            model_defaults = self.get_model_defaults()
            if model_defaults:
                print(f"[Prompt Rewriter] Applying model default sampling: temp={model_defaults.get('temperature')}, top_k={model_defaults.get('top_k')}, top_p={model_defaults.get('top_p')}, min_p={model_defaults.get('min_p')}, repeat_penalty={model_defaults.get('repeat_penalty')}")
                payload["temperature"] = round(float(model_defaults.get("temperature", 0.8)), 4)
                payload["top_k"] = int(model_defaults.get("top_k", 40))
                payload["top_p"] = round(float(model_defaults.get("top_p", 0.95)), 4)
                payload["min_p"] = round(float(model_defaults.get("min_p", 0.05)), 4)
                payload["repeat_penalty"] = round(float(model_defaults.get("repeat_penalty", 1.0)), 4)
            else:
                print("[Prompt Rewriter] Warning: Could not fetch model defaults, using fallback sampling values")
                payload["temperature"] = 0.8
                payload["top_k"] = 40
                payload["top_p"] = 0.95
                payload["min_p"] = 0.05
                payload["repeat_penalty"] = 1.0
            payload["max_tokens"] = int(options.get("max_tokens", 8192)) if options else 8192
        else:
            if options:
                if "temperature" in options:
                    payload["temperature"] = round(float(options["temperature"]), 4)
                if "top_p" in options:
                    payload["top_p"] = round(float(options["top_p"]), 4)
                if "top_k" in options:
                    payload["top_k"] = int(options["top_k"])
                if "min_p" in options:
                    payload["min_p"] = round(float(options["min_p"]), 4)
                if "repeat_penalty" in options:
                    payload["repeat_penalty"] = round(float(options["repeat_penalty"]), 4)
            payload["max_tokens"] = int(options.get("max_tokens", 8192)) if options else 8192

        full_url = f"{self.SERVER_URL}/v1/chat/completions"

        try:
            if _current_model:
                print(f"[Prompt Rewriter] Generating with model: {_current_model} (backend: {_current_backend})")
            
            if show_everything_in_console:
                self._print_debug_header(payload, enable_thinking, use_model_default_sampling, images)

            response = requests.post(
                full_url,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            full_response = ""
            thinking_content = ""
            usage_stats = None
            timings_stats = None
            first_content_received = False
            first_thinking_received = False
            
            # Add these new flags:
            thinking_section_opened = False
            answer_section_opened = False
            
            for line in response.iter_lines():
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except comfy.model_management.InterruptProcessingException:
                    print("[Prompt Rewriter] Generation interrupted by user")
                    response.close()
                    if stop_server_after:
                        self.stop_server()
                    raise
                
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                        
                        if json_str.strip() == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(json_str)

                            if "usage" in chunk:
                                usage_stats = chunk["usage"]
                            if "timings" in chunk:
                                timings_stats = chunk["timings"]

                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                
                                # Handle reasoning/thinking content FIRST
                                reasoning_delta = delta.get('reasoning_content', '')
                                if reasoning_delta:
                                    # Strip leading newlines from first thinking content
                                    if not first_thinking_received:
                                        reasoning_delta = reasoning_delta.lstrip('\n')
                                        first_thinking_received = True
                                    
                                    thinking_content += reasoning_delta
                                    if show_everything_in_console:
                                        if not thinking_section_opened:
                                            print_section("thinking")
                                            thinking_section_opened = True
                                        print(reasoning_delta, end='', flush=True)
                                
                                # Handle answer content
                                content = delta.get('content', '')
                                if content:
                                    # Strip leading newlines from first content only
                                    if not first_content_received:
                                        content = content.lstrip('\n')
                                        first_content_received = True
                                    
                                    full_response += content
                                    if show_everything_in_console:
                                        # Close thinking section if it was open
                                        if thinking_section_opened and not answer_section_opened:
                                            print() 
                                            print()
                                            print_section("final answer")
                                            answer_section_opened = True
                                        elif not answer_section_opened:
                                            # No thinking, just answer
                                            print_section("final answer")
                                            answer_section_opened = True
                                        print(content, end='', flush=True)
                                        
                        except json.JSONDecodeError:
                            pass

            if show_everything_in_console:
                # Close any open sections
                if answer_section_opened:
                    print()  # Add newline BEFORE closing final answer
                elif thinking_section_opened:
                    # Edge case: only thinking, no answer
                    print()  # Add newline BEFORE closing thinking
                
                print()  # Extra newline before token stats
                
                # Use pre-cached token counts (already fetched before generation)
                self._print_token_stats(
                    usage_stats, 
                    cached_token_counts,
                    thinking_content, 
                    full_response, 
                    images,
                    timings_stats
                )

            if not full_response:
                print("[Prompt Rewriter] Warning: Empty response from server")
                full_response = prompt

            print("[Prompt Rewriter] Successfully generated prompt")

            # Store ONLY this result, replacing any previous cache
            self._prompt_cache = {
                "last_key": cache_key,
                "last_result": full_response
            }

            if stop_server_after:
                self.stop_server()

            return (full_response,)  

        except comfy.model_management.InterruptProcessingException:
            raise
        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.text
            except:
                pass
            
            error_msg = f"Error: HTTP {e.response.status_code}"
            if error_body:
                error_msg += f"\nServer response: {error_body[:1000]}"
            
            print(f"[Prompt Rewriter] {error_msg}")
            
            if images and e.response.status_code == 500:
                print(
                    f"When processing images, this can happen if the context window is too small. "
                    f"Consider increasing context_size (current: {context_size})."
                )
            
            return (error_msg,)
        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to server at {full_url}. Server may have crashed."
            print(f"[Prompt Rewriter] {error_msg}")
            return (error_msg,)
        except requests.exceptions.Timeout:
            error_msg = "Error: Request timed out (>300s)"
            print(f"[Prompt Rewriter] {error_msg}")
            return (error_msg,)
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[Prompt Rewriter] {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)

    def _get_token_counts_parallel(self, system_prompt, user_prompt):
        """Get token counts for system and user prompts in parallel using threads"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = {"system": None, "user": None}
        
        def tokenize_system():
            return self.count_tokens(system_prompt)
        
        def tokenize_user():
            return self.count_tokens(user_prompt)
        
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_sys = executor.submit(tokenize_system)
                future_usr = executor.submit(tokenize_user)
                
                results["system"] = future_sys.result(timeout=15)
                results["user"] = future_usr.result(timeout=15)
        except Exception as e:
            print(f"[Prompt Rewriter] Warning: Parallel tokenization failed: {e}")
        
        return results

    def _print_token_stats(self, usage_stats, cached_token_counts, thinking_content, full_response, images, timings_stats=None):
        """Print token statistics using pre-cached counts"""
        
        # Get tokens per second from llama.cpp timings
        tokens_per_sec = None
        if timings_stats:
            tokens_per_sec = timings_stats.get('predicted_per_second')
        
        print("="*60)
        if tokens_per_sec:
            print(f" [Prompt Rewriter] TOKEN USAGE STATISTICS ({tokens_per_sec:.2f} tokens/s)")
        else:
            print(" [Prompt Rewriter] TOKEN USAGE STATISTICS")
        print("="*60)

        total_input = usage_stats.get('prompt_tokens', 0) if usage_stats else 0
        total_output = usage_stats.get('completion_tokens', 0) if usage_stats else 0

        # Use cached token counts - handle None values
        sys_tokens = cached_token_counts.get("system") if cached_token_counts else None
        usr_tokens = cached_token_counts.get("user") if cached_token_counts else None
        
        # Convert None to 0 for arithmetic, but track if we have valid counts
        sys_tokens_val = sys_tokens if sys_tokens is not None else 0
        usr_tokens_val = usr_tokens if usr_tokens is not None else 0
        
        # Image tokens = total input - text tokens
        text_tokens = sys_tokens_val + usr_tokens_val
        image_tokens = max(0, total_input - text_tokens) if images else 0

        # Output token split
        think_len = len(thinking_content) if thinking_content else 0
        ans_len = len(full_response) if full_response else 0
        total_out_len = think_len + ans_len

        if total_output > 0 and total_out_len > 0:
            think_tokens = int(total_output * (think_len / total_out_len))
            ans_tokens = total_output - think_tokens
        else:
            think_tokens = 0
            ans_tokens = 0
        
        # Display with "N/A" if tokenization failed
        if sys_tokens is not None:
            print(f" SYSTEM PROMPT: {sys_tokens:>5} tokens")
        else:
            print(f" SYSTEM PROMPT:   N/A (tokenization failed)")
            
        if usr_tokens is not None:
            print(f" USER PROMPT:   {usr_tokens:>5} tokens")
        else:
            print(f" USER PROMPT:     N/A (tokenization failed)")
            
        if images and image_tokens > 0:
            image_label = "image" if len(images) == 1 else "images"
            print(f" IMAGES:        {image_tokens:>5} tokens ({len(images)} {image_label})")
        print(f" -----------------------------")
        print(f" THINKING:      {think_tokens:>5} tokens")
        print(f" FINAL ANSWER:  {ans_tokens:>5} tokens")
        print(f" -----------------------------")
        print(f" TOTAL:         {total_input + total_output:>5} tokens")
        
        print("="*60 + "\n")


NODE_CLASS_MAPPINGS = {
    "PromptRewriterZ": PromptRewriterZ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRewriterZ": "Prompt Rewriter"
}
