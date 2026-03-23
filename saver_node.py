import os
import re
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from server import PromptServer
import comfy.cli_args

# Counter state per node instance: {uid: counter, (uid, "start"): last_counter_start}
_counters = {}


def _find_max_counter(directory, prefix, ext):
    """Scan directory (and subdirs) for existing files matching prefix_NNNNN.ext
    and return the highest counter found, or -1 if none."""
    pattern = re.compile(
        re.escape(prefix) + r"_(\d{5})" + re.escape(ext) + "$",
        re.IGNORECASE,
    )
    max_counter = -1
    for dirpath, _dirs, filenames in os.walk(directory):
        for f in filenames:
            m = pattern.match(f)
            if m:
                c = int(m.group(1))
                if c > max_counter:
                    max_counter = c
    return max_counter


class BatchImageSaver:
    """Saves up to 5 images into organized subfolders with synchronized filenames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Base folder path where images will be saved",
                }),
                "prefix": ("STRING", {
                    "default": "img",
                    "tooltip": "Filename prefix (e.g. 'img' produces img_00001.png)",
                }),
                "format": (["png", "jpg", "webp"], {
                    "default": "png",
                    "tooltip": "Image format to save",
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Quality for lossy formats (JPEG/WebP). PNG uses compression level instead.",
                }),
                "counter_start": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Starting number for the filename counter",
                }),
            },
            "optional": {
                "filename_override": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "If provided, saves with this exact filename (ignores prefix/counter/format). Connect the 'filename' output from Iterator/Loader.",
                }),
                "subfolder_1": ("STRING", {"default": ""}),
                "subfolder_2": ("STRING", {"default": ""}),
                "subfolder_3": ("STRING", {"default": ""}),
                "subfolder_4": ("STRING", {"default": ""}),
                "subfolder_5": ("STRING", {"default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
                "mask_5": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_filename",)
    FUNCTION = "save_images"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, output_path, **kwargs):
        if not output_path:
            return "output_path is required"
        return True

    @staticmethod
    def _tensor_to_pil(image_tensor, mask_tensor=None):
        """Convert a ComfyUI IMAGE tensor (and optional MASK) to a PIL Image."""
        # IMAGE shape: [B, H, W, C] where C=3 (RGB), values 0.0-1.0
        img_np = image_tensor[0].cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        if mask_tensor is not None:
            # MASK shape: [B, H, W], values 0.0-1.0
            if mask_tensor.ndim == 3:
                mask_np = mask_tensor[0].cpu().numpy()
            elif mask_tensor.ndim == 2:
                mask_np = mask_tensor.cpu().numpy()
            else:
                # Unexpected shape, skip mask
                return Image.fromarray(img_np, mode="RGB")

            # Resize mask if dimensions don't match
            h, w = img_np.shape[:2]
            if mask_np.shape != (h, w):
                mask_pil = Image.fromarray(
                    np.clip(mask_np * 255, 0, 255).astype(np.uint8), mode="L"
                )
                mask_pil = mask_pil.resize((w, h), Image.LANCZOS)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0

            # Invert: ComfyUI mask convention is inverted from alpha
            # mask=0 means visible (alpha=255), mask=1 means masked (alpha=0)
            alpha = np.clip((1.0 - mask_np) * 255.0, 0, 255).astype(np.uint8)
            rgba = np.concatenate([img_np, alpha[:, :, np.newaxis]], axis=-1)
            return Image.fromarray(rgba, mode="RGBA")

        return Image.fromarray(img_np, mode="RGB")

    @staticmethod
    def _get_extension(fmt):
        """Get file extension for the given format."""
        return {"png": ".png", "jpg": ".jpg", "webp": ".webp"}[fmt]

    @staticmethod
    def _build_png_metadata(prompt, extra_pnginfo):
        """Build PngInfo metadata if available and not disabled."""
        if comfy.cli_args.args.disable_metadata:
            return None
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key in extra_pnginfo:
                metadata.add_text(key, json.dumps(extra_pnginfo[key]))
        return metadata

    @staticmethod
    def _save_pil_image(pil_image, filepath, fmt, quality, metadata=None):
        """Save a PIL Image with the appropriate format settings."""
        if fmt == "png":
            save_kwargs = {"format": "PNG", "compress_level": 6}
            if metadata is not None:
                save_kwargs["pnginfo"] = metadata
            pil_image.save(filepath, **save_kwargs)
        elif fmt == "jpg":
            # JPEG doesn't support transparency - convert to RGB
            if pil_image.mode == "RGBA":
                bg = Image.new("RGB", pil_image.size, (255, 255, 255))
                bg.paste(pil_image, mask=pil_image.split()[3])
                pil_image = bg
            elif pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(filepath, format="JPEG", quality=quality)
        elif fmt == "webp":
            pil_image.save(filepath, format="WEBP", quality=quality, lossless=(quality == 100))

    def save_images(self, output_path, prefix, format, quality, counter_start,
                    unique_id=None, prompt=None, extra_pnginfo=None, **kwargs):
        if not output_path:
            raise ValueError("output_path is required")

        resolved_path = os.path.realpath(output_path)
        if not os.path.isdir(resolved_path):
            os.makedirs(resolved_path, exist_ok=True)

        uid = str(unique_id) if unique_id is not None else "default"
        filename_override = kwargs.pop("filename_override", None) or ""
        filename_override = filename_override.strip()

        if filename_override:
            # Use the exact filename provided (from Iterator/Loader output)
            filename = filename_override
            # Detect format from the override extension for save logic
            _, override_ext = os.path.splitext(filename)
            override_ext = override_ext.lower()
            if override_ext in (".jpg", ".jpeg"):
                save_fmt = "jpg"
            elif override_ext in (".webp",):
                save_fmt = "webp"
            else:
                save_fmt = "png"
            metadata = self._build_png_metadata(prompt, extra_pnginfo) if save_fmt == "png" else None
        else:
            # Normal counter-based naming
            ext = self._get_extension(format)
            save_fmt = format

            last_start = _counters.get((uid, "start"))
            if uid not in _counters or last_start != counter_start:
                disk_max = _find_max_counter(resolved_path, prefix, ext)
                counter = max(counter_start, disk_max + 1)
                _counters[uid] = counter
                _counters[(uid, "start")] = counter_start
            else:
                counter = _counters[uid]
                disk_max = _find_max_counter(resolved_path, prefix, ext)
                if disk_max >= counter:
                    counter = disk_max + 1
                    _counters[uid] = counter

            metadata = self._build_png_metadata(prompt, extra_pnginfo) if format == "png" else None
            filename = f"{prefix}_{counter:05d}{ext}"
        saved_files = []
        saved_count = 0

        for i in range(1, 6):
            image = kwargs.get(f"image_{i}")
            if image is None:
                continue

            subfolder = kwargs.get(f"subfolder_{i}", "")
            mask = kwargs.get(f"mask_{i}")

            # Build target directory
            if subfolder:
                # Sanitize: normalize path, remove .. components entirely
                safe_subfolder = subfolder.strip()
                # Use os.path.normpath to collapse redundant separators and ..
                safe_subfolder = os.path.normpath(safe_subfolder)
                # Remove any leading path separators or drive letters
                safe_subfolder = safe_subfolder.lstrip(os.sep).lstrip("/")
                # Remove any remaining .. components
                parts = safe_subfolder.split(os.sep)
                parts = [p for p in parts if p and p != ".."]
                safe_subfolder = os.sep.join(parts) if parts else ""
                if not safe_subfolder:
                    target_dir = resolved_path
                else:
                    target_dir = os.path.join(resolved_path, safe_subfolder)
            else:
                target_dir = resolved_path

            # Verify path stays within output_path BEFORE creating directories
            real_target = os.path.realpath(target_dir)
            if real_target != resolved_path and not real_target.startswith(resolved_path + os.sep):
                continue

            os.makedirs(real_target, exist_ok=True)

            # Convert and save (PNG and WebP support alpha via mask)
            has_mask = mask is not None and save_fmt in ("png", "webp")
            pil_image = self._tensor_to_pil(image, mask if has_mask else None)

            filepath = os.path.join(real_target, filename)
            self._save_pil_image(pil_image, filepath, save_fmt, quality, metadata)

            saved_files.append(f"  [{i}] {os.path.relpath(filepath, resolved_path)}")
            saved_count += 1

        # Only increment counter if at least one image was saved (skip for filename_override)
        if saved_count > 0 and not filename_override:
            _counters[uid] = counter + 1

        # Build info text
        if saved_count > 0:
            info_lines = [f"Saved {saved_count} image(s) as '{filename}'"]
            info_lines.extend(saved_files)
        else:
            info_lines = ["No images connected or saved."]
        info_text = "\n".join(info_lines)

        PromptServer.instance.send_sync("batch_saver.update", {
            "node": uid,
            "filename": filename if saved_count > 0 else "",
            "count": saved_count,
            "info": info_text,
        })

        return {
            "ui": {
                "text": [info_text],
            },
            "result": (filename,),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
