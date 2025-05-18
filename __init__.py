NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from . import nodes_gguf
    
    NODE_CLASS_MAPPINGS = {
        "JJC_JoyCaption_GGUF": nodes_gguf.JoyCaptionGGUF,
        "JJC_JoyCaption_Custom_GGUF": nodes_gguf.JoyCaptionCustomGGUF,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "JJC_JoyCaption_GGUF": "JoyCaption (GGUF)",
        "JJC_JoyCaption_Custom_GGUF": "JoyCaption (Custom GGUF)",
    }
    print("[JoyCaption] GGUF nodes loaded.")

except ImportError:
    print("[JoyCaption] GGUF nodes not available. This usually means 'llama-cpp-python' is not installed or there's an issue in 'nodes_gguf.py'.")
    # Mappings will remain empty if import fails
except Exception as e:
    print(f"[JoyCaption] Error loading GGUF nodes: {e}")
    # Mappings will remain empty if import fails

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
