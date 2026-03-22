from .nodes import ImageIterator
from .saver_node import BatchImageSaver
from .combiner_node import ImageCombiner
from .loader_node import BatchImageLoader

NODE_CLASS_MAPPINGS = {
    "ImageIterator": ImageIterator,
    "BatchImageSaver": BatchImageSaver,
    "ImageCombiner": ImageCombiner,
    "BatchImageLoader": BatchImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIterator": "Image Iterator 🔄",
    "BatchImageSaver": "Batch Image Saver 💾",
    "ImageCombiner": "Image Combiner 🔗",
    "BatchImageLoader": "Batch Image Loader 📂",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
