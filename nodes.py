import os
import json
import threading
import torch
import numpy as np
from PIL import Image, ImageOps
from aiohttp import web
from server import PromptServer

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".tiff", ".tif", ".gif", ".ico",
}

# JavaScript safe integer max (2^53 - 1)
JS_MAX_SAFE_INT = 0x1FFFFFFFFFFFFF


class ImageIterator:
    """Loads images from a folder one at a time, iterating through them sequentially."""

    # Class-level state: keyed by unique_id
    _file_cache = {}  # {unique_id: [sorted list of file paths]}
    _cache_keys = {}  # {unique_id: (folder_path, extensions, sort_by)} for invalidation
    _internal_index = {}  # {unique_id: current server-side index}
    _lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to the folder containing images",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Current image index (auto-incremented after each run)",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": JS_MAX_SAFE_INT,
                    "tooltip": "Seed to trigger re-execution (use randomize)",
                }),
                "file_extensions": ("STRING", {
                    "default": "*.png,*.jpg,*.jpeg,*.webp",
                    "tooltip": "Comma-separated list of extensions to filter (e.g. *.png,*.jpg)",
                }),
                "sort_by": (["alphabetical", "modified_date", "created_date"], {
                    "default": "alphabetical",
                    "tooltip": "How to sort the files in the folder",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "filename", "index", "total")
    FUNCTION = "load_image"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def _parse_extensions(cls, file_extensions):
        """Parse the extensions string into a set of lowercase extensions."""
        exts = set()
        for ext in file_extensions.split(","):
            ext = ext.strip().lower()
            if ext.startswith("*."):
                ext = ext[1:]  # remove the *
            elif ext and not ext.startswith("."):
                ext = "." + ext
            if ext:
                exts.add(ext)
        return exts if exts else SUPPORTED_EXTENSIONS

    @classmethod
    def _validate_folder_path(cls, folder_path):
        """Validate and resolve the folder path to prevent traversal attacks."""
        resolved = os.path.realpath(folder_path)
        if not os.path.isdir(resolved):
            return None
        return resolved

    @classmethod
    def _scan_folder(cls, folder_path, extensions, sort_by):
        """Scan a folder recursively and return a sorted list of image file paths."""
        if not os.path.isdir(folder_path):
            return []

        resolved_root = os.path.realpath(folder_path)
        files = []
        for dirpath, dirnames, filenames in os.walk(resolved_root):
            dirnames.sort()  # Deterministic subdirectory traversal order
            for f in filenames:
                full_path = os.path.join(dirpath, f)
                real_path = os.path.realpath(full_path)
                if not real_path.startswith(resolved_root + os.sep):
                    continue
                _, ext = os.path.splitext(f)
                if ext.lower() in extensions:
                    files.append(real_path)

        if sort_by == "alphabetical":
            # Sort by full path for deterministic ordering across subdirectories
            files.sort(key=lambda p: p.lower())
        elif sort_by == "modified_date":
            files.sort(key=lambda p: (os.path.getmtime(p), p.lower()))
        elif sort_by == "created_date":
            files.sort(key=lambda p: (os.path.getctime(p), p.lower()))

        return files

    @classmethod
    def _get_files(cls, unique_id, folder_path, file_extensions, sort_by):
        """Get file list from cache or scan the folder."""
        extensions = cls._parse_extensions(file_extensions)
        try:
            folder_mtime = os.path.getmtime(folder_path)
        except OSError:
            folder_mtime = 0
        cache_key = (folder_path, frozenset(extensions), sort_by, folder_mtime)

        with cls._lock:
            if unique_id in cls._cache_keys and cls._cache_keys[unique_id] == cache_key:
                return list(cls._file_cache.get(unique_id, []))

            files = cls._scan_folder(folder_path, extensions, sort_by)
            cls._file_cache[unique_id] = files
            cls._cache_keys[unique_id] = cache_key
            return list(files)

    @classmethod
    def _invalidate_cache(cls, unique_id):
        """Force re-scan and reset index on next execution."""
        with cls._lock:
            cls._file_cache.pop(unique_id, None)
            cls._cache_keys.pop(unique_id, None)
            cls._internal_index.pop(unique_id, None)

    @staticmethod
    def _load_image(image_path):
        """Load an image file and return a ComfyUI-compatible IMAGE tensor."""
        with Image.open(image_path) as i:
            i = ImageOps.exif_transpose(i)
            if i.mode == "I":
                # 32-bit integer mode: convert to float first for safe normalization
                arr = np.array(i).astype(np.float32) / 65535.0
                image = np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr
            else:
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def load_image(self, folder_path, index, seed, file_extensions, sort_by, unique_id=None):
        resolved_path = self._validate_folder_path(folder_path)
        if not resolved_path:
            raise ValueError(f"Invalid folder path: {folder_path}")

        uid = str(unique_id) if unique_id is not None else "default"
        files = self._get_files(uid, resolved_path, file_extensions, sort_by)

        if not files:
            raise ValueError(f"No images found in '{folder_path}' with extensions '{file_extensions}'")

        total = len(files)

        # Server-side index management: fully independent of frontend widget
        # The widget "index" is ONLY used as starting point on first run.
        # After that, the server auto-increments independently.
        # Use Reset button to restart from 0.
        with self._lock:
            if uid not in self._internal_index:
                # First run: use widget value as starting point
                self._internal_index[uid] = index % total

            current_index = self._internal_index[uid] % total
            next_index = (current_index + 1) % total

            # Auto-increment for next execution
            self._internal_index[uid] = next_index

        current_file = files[current_index]

        # Fallback: if cached file was deleted, re-scan and retry
        if not os.path.isfile(current_file):
            self._invalidate_cache(uid)
            files = self._get_files(uid, resolved_path, file_extensions, sort_by)
            if not files:
                raise ValueError(f"No images found in '{folder_path}'")
            total = len(files)
            with self._lock:
                self._internal_index[uid] = self._internal_index.get(uid, 0) % total
                current_index = self._internal_index[uid]
                next_index = (current_index + 1) % total
                self._internal_index[uid] = next_index
            current_file = files[current_index]

        # Use relative path for display (distinguishes files in different subdirs)
        current_filename = os.path.relpath(current_file, resolved_path)
        next_filename = os.path.relpath(files[next_index % len(files)], resolved_path)

        image = self._load_image(current_file)

        info_text = f"[{current_index + 1}/{total}] {current_filename}"
        next_text = f"Next: {next_filename}" if total > 1 else "Next: (loop)"

        PromptServer.instance.send_sync("image_iterator.update", {
            "node": unique_id,
            "current_index": current_index,
            "next_index": next_index,
            "total": total,
            "current_file": current_filename,
            "next_file": next_filename,
            "info": info_text,
            "next_info": next_text,
        })

        return {
            "ui": {
                "text": [info_text, next_text],
            },
            "result": (image, current_filename, current_index, total),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, index, seed, file_extensions, sort_by, unique_id=None):
        if not folder_path:
            return "folder_path is required"
        resolved = os.path.realpath(folder_path)
        if not os.path.isdir(resolved):
            return f"Folder not found: {folder_path}"
        return True

    @classmethod
    def IS_CHANGED(cls, folder_path, index, seed, file_extensions, sort_by, unique_id=None):
        # Always re-execute: server-side index auto-increments independently
        return float("NaN")


# --- Custom API Routes ---

@PromptServer.instance.routes.post("/image_iterator/reset")
async def reset_iterator(request):
    """Reset the iterator: clear cache and return index 0."""
    try:
        data = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    unique_id = str(data.get("node_id", ""))

    if unique_id:
        ImageIterator._invalidate_cache(unique_id)

    return web.json_response({"status": "ok", "index": 0})


@PromptServer.instance.routes.get("/image_iterator/info/{node_id}")
async def get_iterator_info(request):
    """Get current cached file count for a node."""
    node_id = request.match_info["node_id"]

    with ImageIterator._lock:
        files = ImageIterator._file_cache.get(node_id, [])
        total = len(files)
        filenames = [os.path.basename(f) for f in files]

    return web.json_response({
        "total": total,
        "files": filenames,
    })
