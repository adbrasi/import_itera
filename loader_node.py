import os
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


class BatchImageLoader:
    """Loads images from multiple subfolders by synchronized filename.

    The inverse of BatchImageSaver: given an input_path and up to 5 subfolder
    names, iterates through files found in any subfolder (union), loading the
    same filename from each subfolder on each execution.
    """

    _file_cache = {}    # {uid: {basename: {subfolder_name: full_path}}}
    _file_list = {}     # {uid: [sorted list of basenames]}
    _cache_keys = {}    # {uid: cache_key tuple}
    _internal_index = {}
    _lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "default": "",
                    "tooltip": "Base folder path containing the subfolders",
                }),
                "subfolder_1": ("STRING", {
                    "default": "",
                    "tooltip": "First subfolder name (required)",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Starting index (only used on first run, then auto-increments)",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": JS_MAX_SAFE_INT,
                    "tooltip": "Seed to trigger re-execution (use randomize)",
                }),
                "file_extensions": ("STRING", {
                    "default": "*.png,*.jpg,*.jpeg,*.webp",
                    "tooltip": "Comma-separated extensions to filter",
                }),
                "sort_by": (["alphabetical", "modified_date", "created_date"], {
                    "default": "alphabetical",
                }),
            },
            "optional": {
                "subfolder_2": ("STRING", {"default": ""}),
                "subfolder_3": ("STRING", {"default": ""}),
                "subfolder_4": ("STRING", {"default": ""}),
                "subfolder_5": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image_1", "image_2", "image_3", "image_4", "image_5",
                    "filename", "index", "total")
    FUNCTION = "load_batch"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def _parse_extensions(cls, file_extensions):
        """Parse extensions string into a set of lowercase extensions."""
        exts = set()
        for ext in file_extensions.split(","):
            ext = ext.strip().lower()
            if ext.startswith("*."):
                ext = ext[1:]
            elif ext and not ext.startswith("."):
                ext = "." + ext
            if ext:
                exts.add(ext)
        return exts if exts else SUPPORTED_EXTENSIONS

    @classmethod
    def _get_subfolder_mtimes(cls, input_path, subfolders):
        """Get max mtime across input_path and all active subfolders."""
        resolved = os.path.realpath(input_path)
        mtimes = []
        try:
            mtimes.append(os.path.getmtime(resolved))
        except OSError:
            pass
        for sf in subfolders:
            sf_path = os.path.join(resolved, sf)
            try:
                mtimes.append(os.path.getmtime(sf_path))
            except OSError:
                pass
        return max(mtimes) if mtimes else 0

    @classmethod
    def _scan_subfolders(cls, input_path, subfolders, extensions, sort_by):
        """Scan all subfolders and build a union map of {basename: {subfolder: path}}."""
        resolved_root = os.path.realpath(input_path)
        file_map = {}

        for sf_name in subfolders:
            sf_path = os.path.join(resolved_root, sf_name)
            sf_real = os.path.realpath(sf_path)

            if sf_real != resolved_root and not sf_real.startswith(resolved_root + os.sep):
                continue
            if not os.path.isdir(sf_real):
                continue

            for f in os.listdir(sf_real):
                full_path = os.path.join(sf_real, f)
                if not os.path.isfile(full_path):
                    continue
                _, ext = os.path.splitext(f)
                if ext.lower() not in extensions:
                    continue

                if f not in file_map:
                    file_map[f] = {}
                file_map[f][sf_name] = full_path

        basenames = list(file_map.keys())
        if sort_by == "alphabetical":
            basenames.sort(key=lambda b: b.lower())
        elif sort_by == "modified_date":
            def get_mtime(b):
                return min(os.path.getmtime(p) for p in file_map[b].values())
            basenames.sort(key=get_mtime)
        elif sort_by == "created_date":
            def get_ctime(b):
                return min(os.path.getctime(p) for p in file_map[b].values())
            basenames.sort(key=get_ctime)

        return file_map, basenames

    @classmethod
    def _get_files(cls, uid, input_path, subfolders, file_extensions, sort_by):
        """Get file map from cache or scan."""
        extensions = cls._parse_extensions(file_extensions)
        max_mtime = cls._get_subfolder_mtimes(input_path, subfolders)
        cache_key = (input_path, tuple(subfolders), frozenset(extensions), sort_by, max_mtime)

        with cls._lock:
            if uid in cls._cache_keys and cls._cache_keys[uid] == cache_key:
                return cls._file_cache.get(uid, {}), list(cls._file_list.get(uid, []))

            file_map, basenames = cls._scan_subfolders(input_path, subfolders, extensions, sort_by)
            cls._file_cache[uid] = file_map
            cls._file_list[uid] = basenames
            cls._cache_keys[uid] = cache_key
            return file_map, list(basenames)

    @classmethod
    def _invalidate_cache_unsafe(cls, uid):
        """Clear cache without acquiring lock. Caller must hold the lock."""
        cls._file_cache.pop(uid, None)
        cls._file_list.pop(uid, None)
        cls._cache_keys.pop(uid, None)
        cls._internal_index.pop(uid, None)

    @classmethod
    def _invalidate_cache(cls, uid):
        """Force re-scan and reset index (acquires lock)."""
        with cls._lock:
            cls._invalidate_cache_unsafe(uid)

    @staticmethod
    def _load_image(image_path):
        """Load an image and return a ComfyUI IMAGE tensor [1, H, W, 3]."""
        with Image.open(image_path) as i:
            i = ImageOps.exif_transpose(i)
            if i.mode == "I":
                arr = np.array(i).astype(np.float32) / 65535.0
                image = np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr
            else:
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image)[None,]

    @staticmethod
    def _empty_image():
        """Return a 1x1 fallback image [1, 1, 1, 3]."""
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)

    def load_batch(self, input_path, subfolder_1, index, seed, file_extensions,
                   sort_by, unique_id=None, **kwargs):
        resolved_path = os.path.realpath(input_path)
        if not os.path.isdir(resolved_path):
            raise ValueError(f"Invalid input path: {input_path}")

        subfolders = [subfolder_1] if subfolder_1 else []
        for i in range(2, 6):
            sf = kwargs.get(f"subfolder_{i}", "")
            if sf:
                subfolders.append(sf)

        if not subfolders:
            raise ValueError("At least one subfolder is required")

        uid = str(unique_id) if unique_id is not None else "default"
        file_map, basenames = self._get_files(uid, resolved_path, subfolders, file_extensions, sort_by)

        if not basenames:
            raise ValueError(f"No images found in subfolders of '{input_path}'")

        total = len(basenames)

        # Server-side index: find next valid basename
        with self._lock:
            if uid not in self._internal_index:
                self._internal_index[uid] = index % total

            start = self._internal_index[uid] % total
            current_index = start
            current_basename = None
            attempts = 0

            while attempts < total:
                candidate = basenames[current_index % total]
                has_file = any(
                    os.path.isfile(fpath)
                    for fpath in file_map.get(candidate, {}).values()
                )
                if has_file:
                    current_basename = candidate
                    current_index = current_index % total
                    break
                current_index += 1
                attempts += 1

            if current_basename is None:
                # All cached files gone — clear cache without deadlock
                self._invalidate_cache_unsafe(uid)
                raise ValueError("No valid image files found on disk")

            self._internal_index[uid] = (current_index + 1) % total

        # Load images from each subfolder (outside lock)
        images = []
        for sf_name in subfolders:
            fpath = file_map.get(current_basename, {}).get(sf_name)
            if fpath and os.path.isfile(fpath):
                images.append(self._load_image(fpath))
            else:
                images.append(self._empty_image())

        while len(images) < 5:
            images.append(self._empty_image())

        info_text = f"[{current_index + 1}/{total}] {current_basename}"
        found_in = [sf for sf in subfolders if file_map.get(current_basename, {}).get(sf)]
        next_idx = (current_index + 1) % total
        next_name = basenames[next_idx] if total > 1 else current_basename
        next_text = f"Next: {next_name} | Found in: {', '.join(found_in)}"

        PromptServer.instance.send_sync("batch_loader.update", {
            "node": unique_id,
            "info": info_text,
            "next_info": next_text,
        })

        return {
            "ui": {"text": [info_text, next_text]},
            "result": (images[0], images[1], images[2], images[3], images[4],
                       current_basename, current_index, total),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_path, subfolder_1, **kwargs):
        if not input_path:
            return "input_path is required"
        if not subfolder_1:
            return "subfolder_1 is required"
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# --- Custom API Routes ---

@PromptServer.instance.routes.post("/batch_loader/reset")
async def reset_batch_loader(request):
    """Reset the loader: clear cache and index."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    unique_id = str(data.get("node_id", ""))
    if unique_id:
        BatchImageLoader._invalidate_cache(unique_id)

    return web.json_response({"status": "ok"})
