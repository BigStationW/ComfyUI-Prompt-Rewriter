from .prompt_generator import PromptGeneratorZ
from .prompt_generator_options import PromptGenOptionsZ

NODE_CLASS_MAPPINGS = {
    "PromptGeneratorZ": PromptGeneratorZ,
    "PromptGenOptionsZ": PromptGenOptionsZ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGeneratorZ": "Prompt Generator",
    "PromptGenOptionsZ": "Prompt Generator Options"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
