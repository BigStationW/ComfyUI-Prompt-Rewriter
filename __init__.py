from .prompt_rewriter import PromptRewriterZ
from .prompt_rewriter_options import PromptRewriterOptionsZ

NODE_CLASS_MAPPINGS = {
    "PromptRewriterZ": PromptRewriterZ,
    "PromptRewriterOptionsZ": PromptRewriterOptionsZ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRewriterZ": "Prompt Rewriter",
    "PromptRewriterOptionsZ": "Prompt Rewriter Options"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
