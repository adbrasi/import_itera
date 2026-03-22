import torch


class ImageCombiner:
    """Combines up to 5 transparent images (from background removal models) into one.

    Uses union/max-alpha logic: at each pixel, the layer with the strongest
    alpha (best capture) wins. Masks are unioned — if ANY mask says "subject here",
    the pixel is kept. A good mask protects against errors in other masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "mask_1": ("MASK",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "image_3": ("IMAGE",),
                "mask_3": ("MASK",),
                "image_4": ("IMAGE",),
                "mask_4": ("MASK",),
                "image_5": ("IMAGE",),
                "mask_5": ("MASK",),
                "final_mask": ("MASK", {
                    "tooltip": "Optional final mask: areas where mask=1 will be forced to background color",
                }),
                "background_color": (["white", "black", "transparent"], {
                    "default": "white",
                    "tooltip": "Background color where no image has content",
                }),
                "invert_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert input masks (if your masks have subject=1 instead of ComfyUI standard subject=0)",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("combined_image", "combined_mask")
    FUNCTION = "combine"
    CATEGORY = "image"

    @staticmethod
    def _resize_to(tensor, target_h, target_w, channels=3):
        """Resize a tensor to target dimensions."""
        if channels == 3:
            # [H,W,3] -> [1,3,H,W] -> interpolate -> [H,W,3]
            t = tensor.permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            return t.squeeze(0).permute(1, 2, 0)
        else:
            # [H,W] -> [1,1,H,W] -> interpolate -> [H,W]
            t = tensor.unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            return t.squeeze(0).squeeze(0)

    @staticmethod
    def _combine_union_max(layers):
        """Combine layers using union of masks + max-alpha color selection.

        For each pixel:
        - Alpha (visibility) = max alpha across all layers (union)
        - Color = from the layer with the highest alpha (best capture)

        A good mask in layer 1 protects against errors in layer 2.

        layers: list of (image[H,W,3], alpha[H,W]) tuples, float32 0-1
        Returns: (combined_rgb[H,W,3], combined_alpha[H,W])
        """
        colors = torch.stack([img for img, _ in layers], dim=0)   # [N, H, W, 3]
        alphas = torch.stack([a for _, a in layers], dim=0)       # [N, H, W]

        # Union alpha: take the MAX across all layers at each pixel
        # If ANY layer says "subject here" (alpha > 0), the pixel is kept
        union_alpha, best_idx = alphas.max(dim=0)   # both [H, W]

        # Color: use the layer with the highest alpha (best capture)
        best_idx_expanded = best_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, 3)
        out_color = torch.gather(colors, 0, best_idx_expanded).squeeze(0)  # [H, W, 3]

        return out_color, union_alpha

    @staticmethod
    def _apply_background(color, alpha, bg_type):
        """Composite the result over the chosen background."""
        alpha_3d = alpha.unsqueeze(-1)  # [H, W, 1]

        if bg_type == "transparent":
            return color, alpha
        elif bg_type == "black":
            bg = torch.zeros_like(color)
        else:  # white
            bg = torch.ones_like(color)

        # Standard "over" compositing onto opaque background
        final = color * alpha_3d + bg * (1.0 - alpha_3d)
        return final.clamp(0, 1), alpha

    def combine(self, image_1, mask_1, unique_id=None, **kwargs):
        # Collect all connected image+mask pairs
        pairs = [(image_1, mask_1)]
        for i in range(2, 6):
            img = kwargs.get(f"image_{i}")
            msk = kwargs.get(f"mask_{i}")
            if img is not None and msk is not None:
                pairs.append((img, msk))

        background_color = kwargs.get("background_color", "white")
        invert_masks = kwargs.get("invert_masks", False)
        final_mask_input = kwargs.get("final_mask")

        # Use first image dimensions as reference
        ref_img = pairs[0][0]
        target_h, target_w = ref_img.shape[1], ref_img.shape[2]
        batch_size = ref_img.shape[0]

        results_rgb = []
        results_mask = []

        for b in range(batch_size):
            layers = []
            for img, msk in pairs:
                b_idx = min(b, img.shape[0] - 1)
                m_idx = min(b, msk.shape[0] - 1)
                single_img = img[b_idx]       # [H, W, 3]
                single_mask = msk[m_idx]      # [H, W]

                # Convert mask to alpha
                # ComfyUI standard: mask=0 visible, mask=1 transparent
                # Alpha: 0=transparent, 1=opaque
                if invert_masks:
                    single_alpha = single_mask  # mask already has subject=1
                else:
                    single_alpha = 1.0 - single_mask  # standard ComfyUI

                # Resize if needed
                if single_img.shape[0] != target_h or single_img.shape[1] != target_w:
                    single_img = self._resize_to(single_img, target_h, target_w, 3)
                    single_alpha = self._resize_to(single_alpha, target_h, target_w, 0)

                layers.append((single_img, single_alpha))

            # Combine: union of masks, max-alpha color selection
            combined_color, combined_alpha = self._combine_union_max(layers)

            # Apply optional final mask
            if final_mask_input is not None:
                fm_idx = min(b, final_mask_input.shape[0] - 1)
                fm = final_mask_input[fm_idx]

                if invert_masks:
                    fm_alpha = fm
                else:
                    fm_alpha = 1.0 - fm

                if fm.shape[0] != target_h or fm.shape[1] != target_w:
                    fm_alpha = self._resize_to(fm_alpha, target_h, target_w, 0)

                # Where final_mask says "transparent", force alpha to 0
                combined_alpha = combined_alpha * fm_alpha

            # Apply background
            final_color, final_alpha = self._apply_background(
                combined_color, combined_alpha, background_color
            )

            results_rgb.append(final_color)
            results_mask.append(1.0 - final_alpha)  # Back to ComfyUI mask convention

        out_image = torch.stack(results_rgb, dim=0)
        out_mask = torch.stack(results_mask, dim=0)

        info = f"Combined {len(pairs)} layer(s) [{target_w}x{target_h}] bg={background_color}"

        return {
            "ui": {"text": [info]},
            "result": (out_image, out_mask),
        }
