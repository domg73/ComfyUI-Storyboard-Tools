from .storyboard_grid_generator import StoryboardGridGenerator
from .storyboard_panel_crop import StoryboardPanelCrop
from .image_contrast_adjuster import ImageContrastAdjuster

# Register custom node classes into the ComfyUI pipeline backend environment
NODE_CLASS_MAPPINGS = {
    "StoryboardGridGenerator": StoryboardGridGenerator,
    "StoryboardPanelCrop": StoryboardPanelCrop,
    "ImageContrastAdjuster": ImageContrastAdjuster
}

# Display names mapped directly within the ComfyUI user interface menus
NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardGridGenerator": "Storyboard Grid Generator",
    "StoryboardPanelCrop": "Storyboard Panel Crop",
    "ImageContrastAdjuster": "Image Contrast Adjuster"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
