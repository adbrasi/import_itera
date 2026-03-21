from .nodes import ImageIterator

NODE_CLASS_MAPPINGS = {
    "ImageIterator": ImageIterator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIterator": "Image Iterator 🔄",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
