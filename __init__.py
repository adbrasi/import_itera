from .nodes import ImageIterator
from .saver_node import BatchImageSaver

NODE_CLASS_MAPPINGS = {
    "ImageIterator": ImageIterator,
    "BatchImageSaver": BatchImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIterator": "Image Iterator 🔄",
    "BatchImageSaver": "Batch Image Saver 💾",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
