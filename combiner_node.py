import torch
import numpy as np


class ImageCombiner:
    """Combines up to 5 transparent images into one using alpha-weighted average.

    Each image contributes proportionally to its alpha (opacity) at each pixel.
    No layer takes priority over another — all are treated equally.
    The result is composited over a white background by default.
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
                "blend_mode": (["alpha_weighted_average", "max_alpha_priority"], {
                    "default": "alpha_weighted_average",
                    "tooltip": "alpha_weighted_average: equal contribution. max_alpha_priority: pixel with highest alpha wins.",
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
    def _ensure_same_size(images_and_alphas, target_h, target_w):
        """Resize all images and alphas to target dimensions if needed."""
        result = []
        for img, alpha in images_and_alphas:
            h, w = img.shape[0], img.shape[1]
            if h != target_h or w != target_w:
                # Resize image [H,W,3] -> [1,3,H,W] for interpolate, then back
                img_t = img.permute(2, 0, 1).unsqueeze(0)
                img_t = torch.nn.functional.interpolate(
                    img_t, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                img = img_t.squeeze(0).permute(1, 2, 0)

                # Resize alpha [H,W] -> [1,1,H,W] for interpolate, then back
                alpha_t = alpha.unsqueeze(0).unsqueeze(0)
                alpha_t = torch.nn.functional.interpolate(
                    alpha_t, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                alpha = alpha_t.squeeze(0).squeeze(0)

            result.append((img, alpha))
        return result

    @staticmethod
    def _alpha_weighted_average(layers):
        """Combine layers using alpha-weighted average (equal contribution).

        layers: list of (image[H,W,3], alpha[H,W]) tuples, float32 0-1
        Returns: (combined_rgb[H,W,3], combined_alpha[H,W])
        """
        # Stack all layers
        colors = torch.stack([img for img, _ in layers], dim=0)   # [N, H, W, 3]
        alphas = torch.stack([a for _, a in layers], dim=0)       # [N, H, W]
        alphas_4d = alphas.unsqueeze(-1)                           # [N, H, W, 1]

        # Weighted sum: each color contributes proportional to its alpha
        weighted_sum = (colors * alphas_4d).sum(dim=0)   # [H, W, 3]
        alpha_sum = alphas_4d.sum(dim=0)                  # [H, W, 1]

        # Avoid division by zero
        safe_denom = torch.where(alpha_sum > 1e-7, alpha_sum, torch.ones_like(alpha_sum))
        out_color = (weighted_sum / safe_denom).clamp(0, 1)

        # Output alpha: clamp sum to 1 (opaque where ANY layer contributed)
        out_alpha = alpha_sum.squeeze(-1).clamp(0, 1)    # [H, W]

        return out_color, out_alpha

    @staticmethod
    def _max_alpha_priority(layers):
        """Combine layers using max-alpha priority (pixel with highest alpha wins).

        layers: list of (image[H,W,3], alpha[H,W]) tuples, float32 0-1
        Returns: (combined_rgb[H,W,3], combined_alpha[H,W])
        """
        colors = torch.stack([img for img, _ in layers], dim=0)   # [N, H, W, 3]
        alphas = torch.stack([a for _, a in layers], dim=0)       # [N, H, W]

        # Find which layer has the highest alpha at each pixel
        max_alpha, max_idx = alphas.max(dim=0)   # both [H, W]

        # Gather the color from the layer with highest alpha
        max_idx_expanded = max_idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, 3)
        out_color = torch.gather(colors, 0, max_idx_expanded).squeeze(0)  # [H, W, 3]

        return out_color, max_alpha

    @staticmethod
    def _apply_background(color, alpha, bg_type):
        """Composite the result over the chosen background.

        Returns: (final_image[H,W,3], final_alpha[H,W])
        """
        alpha_3d = alpha.unsqueeze(-1)  # [H, W, 1]

        if bg_type == "transparent":
            # Return as-is; alpha is preserved in the mask output
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
        blend_mode = kwargs.get("blend_mode", "alpha_weighted_average")
        final_mask_input = kwargs.get("final_mask")

        # Use first image dimensions as reference
        ref_img = pairs[0][0]
        target_h, target_w = ref_img.shape[1], ref_img.shape[2]
        batch_size = ref_img.shape[0]

        results_rgb = []
        results_mask = []

        for b in range(batch_size):
            # Extract single batch items and convert masks to alpha
            layers = []
            for img, msk in pairs:
                # Handle batch dimension
                b_idx = min(b, img.shape[0] - 1)
                m_idx = min(b, msk.shape[0] - 1)
                single_img = img[b_idx]   # [H, W, 3]
                # ComfyUI mask convention: 0=visible, 1=transparent
                # Convert to alpha: alpha = 1 - mask
                single_alpha = 1.0 - msk[m_idx]  # [H, W]
                layers.append((single_img, single_alpha))

            # Ensure all layers have the same size
            layers = self._ensure_same_size(layers, target_h, target_w)

            # Combine based on blend mode
            if blend_mode == "max_alpha_priority":
                combined_color, combined_alpha = self._max_alpha_priority(layers)
            else:
                combined_color, combined_alpha = self._alpha_weighted_average(layers)

            # Apply optional final mask (force background where final_mask=1)
            if final_mask_input is not None:
                fm_idx = min(b, final_mask_input.shape[0] - 1)
                fm = final_mask_input[fm_idx]  # [H, W], ComfyUI convention

                # Resize if needed
                if fm.shape[0] != target_h or fm.shape[1] != target_w:
                    fm_t = fm.unsqueeze(0).unsqueeze(0)
                    fm_t = torch.nn.functional.interpolate(
                        fm_t, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                    fm = fm_t.squeeze(0).squeeze(0)

                # Where final_mask=1 (masked/transparent), force alpha to 0
                final_alpha_mask = 1.0 - fm  # Convert to alpha
                combined_alpha = combined_alpha * final_alpha_mask

            # Apply background
            final_color, final_alpha = self._apply_background(
                combined_color, combined_alpha, background_color
            )

            results_rgb.append(final_color)
            # Convert alpha back to ComfyUI mask convention
            results_mask.append(1.0 - final_alpha)

        out_image = torch.stack(results_rgb, dim=0)   # [B, H, W, 3]
        out_mask = torch.stack(results_mask, dim=0)    # [B, H, W]

        info = f"Combined {len(pairs)} layer(s) [{target_w}x{target_h}] mode={blend_mode} bg={background_color}"

        return {
            "ui": {"text": [info]},
            "result": (out_image, out_mask),
        }
