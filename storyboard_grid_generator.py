import torch

class StoryboardGridGenerator:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1152, "min": 64, "max": 8192, "step": 16}),
                "num_panels": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "border_thickness": ("INT", {"default": 4, "min": 0, "max": 64, "step": 2}),
                "background_gray_value": ("INT", {"default": 25, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "invert_colors": ("BOOLEAN", {"default": False}),
                "transparent_cells": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_storyboard_grid"
    CATEGORY = "utils/storyboard"

    def generate_storyboard_grid(self, width, height, num_panels, border_thickness, background_gray_value, invert_colors, transparent_cells):
        if num_panels <= 1: cols, rows = 1, 1
        elif num_panels == 2: cols, rows = 2, 1
        elif num_panels <= 4: cols, rows = 2, 2
        elif num_panels <= 6: cols, rows = 3, 2
        elif num_panels <= 8: cols, rows = 4, 2
        elif num_panels <= 9: cols, rows = 3, 3
        else: cols, rows = 4, 3

        # Base layout initialization: 1.0 represents the default white grid borders
        grid_map = torch.ones((height, width), dtype=torch.float32)

        bg_float = background_gray_value / 255.0

        cell_w = width / cols
        cell_h = height / rows
        half_border = border_thickness / 2

        for r in range(rows):
            for c in range(cols):
                panel_idx = r * cols + c
                if panel_idx >= num_panels:
                    continue

                x_start = int(round(c * cell_w + half_border))
                y_start = int(round(r * cell_h + half_border))
                x_end = int(round((c + 1) * cell_w - half_border))
                y_end = int(round((r + 1) * cell_h - half_border))

                if c == 0: x_start = 0
                if r == 0: y_start = 0
                if c == cols - 1: x_end = width
                if r == rows - 1: y_end = height

                grid_map[y_start:y_end, x_start:x_end] = 0.0

        final_grid = torch.where(grid_map == 1.0, torch.tensor(1.0, dtype=torch.float32), torch.tensor(bg_float, dtype=torch.float32))

        if invert_colors:
            final_grid = 1.0 - final_grid

        # Create standard 3-channel RGB baseline tensor [H, W, 3]
        rgb_tensor = final_grid.unsqueeze(-1).repeat(1, 1, 3)

        if transparent_cells:
            # Alpha Channel Generation: Borders are fully opaque (1.0), internal cells are fully transparent (0.0)
            # Inverting the colors will symmetrically flip the opacity structure mapping if required
            alpha_channel = torch.where(grid_map == 1.0, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))
            if invert_colors:
                alpha_channel = 1.0 - alpha_channel
                
            # Pack as a standard 4-channel RGBA tensor structure [1, H, W, 4]
            rgba_tensor = torch.cat([rgb_tensor, alpha_channel.unsqueeze(-1)], dim=-1)
            grid_image_out = rgba_tensor.unsqueeze(0)
        else:
            # Standard 3-channel RGB layout packed to ComfyUI environment dimensions [1, H, W, 3]
            grid_image_out = rgb_tensor.unsqueeze(0)

        return (grid_image_out,)
