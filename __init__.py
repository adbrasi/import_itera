from .nodes import ImageIterator
from .saver_node import BatchImageSaver
from .combiner_node import ImageCombiner

NODE_CLASS_MAPPINGS = {
    "ImageIterator": ImageIterator,
    "BatchImageSaver": BatchImageSaver,
    "ImageCombiner": ImageCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIterator": "Image Iterator 🔄",
    "BatchImageSaver": "Batch Image Saver 💾",
    "ImageCombiner": "Image Combiner 🔗",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
