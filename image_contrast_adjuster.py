import torch

class ImageContrastAdjuster:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Scaled slider: 1.0 is original. Maximum is capped at 1.5 with very fine steps (0.001)
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.001, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "adjust_contrast"
    CATEGORY = "utils/storyboard"

    def adjust_contrast(self, image, contrast):
        # Midpoint luminance reference for linear normalization
        midpoint = 0.5
        
        # Attenuate the scaling factor to make the slider curve smoother and softer
        # This prevents shadows from clipping aggressively into pure black
        soft_contrast = 1.0 + (contrast - 1.0) * 0.25
        
        # Apply the softened contrast formula across the entire pixel tensor
        adjusted_image = (image - midpoint) * soft_contrast + midpoint
        
        # Clamp bounds strictly to preserve safe [0.0, 1.0] image data compliance
        adjusted_image = torch.clamp(adjusted_image, 0.0, 1.0)
        
        return (adjusted_image,)
