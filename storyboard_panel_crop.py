import torch
import math

class StoryboardPanelCrop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "shave_edges_pixels": ("INT", {"default": 8, "min": 0, "max": 50, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "target_aspect_ratio": ([
                    "original_proportions", 
                    "1:1 (Square)", 
                    "16:9 (Widescreen)", 
                    "3:4 (Vertical)", 
                    "9:16 (Vertical)", 
                    "21:9 (Ultrawide)"
                ], {"default": "16:9 (Widescreen)"}),
                "padding_mode": (["center_crop_zoom", "letterbox_black"], {"default": "center_crop_zoom"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "grid_preview")
    FUNCTION = "process_image"
    CATEGORY = "image/storyboard"

    def process_image(self, image, columns, rows, shave_edges_pixels, offset_y, offset_x, target_aspect_ratio, padding_mode):
        batch_size, height, width, channels = image.shape
        
        # Parallel matrix grid preview generation
        preview = image.clone()
        if shave_edges_pixels > 0:
            # Global outer boundaries shave preview (Yellow allocation)
            preview[:, :shave_edges_pixels, :, 0:2] = 1.0; preview[:, :shave_edges_pixels, :, 2] = 0.0
            preview[:, -shave_edges_pixels:, :, 0:2] = 1.0; preview[:, -shave_edges_pixels:, :, 2] = 0.0
            preview[:, :, :shave_edges_pixels, 0:2] = 1.0; preview[:, :, :shave_edges_pixels, 2] = 0.0
            preview[:, :, -shave_edges_pixels:, 0:2] = 1.0; preview[:, :, -shave_edges_pixels:, 2] = 0.0

        segment_w = width / columns
        segment_h = height / rows

        # Drawing the bidirectional matrix preview grid sequence
        for r in range(rows):
            for c in range(columns):
                start_x = int(round(c * segment_w)) + offset_x
                end_x = int(round((c + 1) * segment_w)) + offset_x
                start_y = int(round(r * segment_h)) + offset_y
                end_y = int(round((r + 1) * segment_h)) + offset_y

                if c == columns - 1: end_x = width
                if r == rows - 1: end_y = height

                # Clamp coordinates for drawing safety to prevent negative/out-of-bound indices
                draw_start_x = max(0, min(width, start_x))
                draw_end_x = max(0, min(width, end_x))
                draw_start_y = max(0, min(height, start_y))
                draw_end_y = max(0, min(height, end_y))

                # Main vertical splitting lines allocation (Red)
                if 0 <= draw_start_x < width:
                    preview[:, :, max(0, draw_start_x-1):min(width, draw_start_x+2), 0] = 1.0
                    preview[:, :, max(0, draw_start_x-1):min(width, draw_start_x+2), 1:] = 0.0
                # Main horizontal splitting lines allocation (Red)
                if 0 <= draw_start_y < height:
                    preview[:, max(0, draw_start_y-1):min(height, draw_start_y+2), :, 0] = 1.0
                    preview[:, max(0, draw_start_y-1):min(height, draw_start_y+2), :, 1:] = 0.0

                # Slices inner safety boundaries shave preview mapping (Cyan)
                if shave_edges_pixels > 0:
                    if (draw_end_x - draw_start_x) > (shave_edges_pixels * 2) and (draw_end_y - draw_start_y) > (shave_edges_pixels * 2):
                        l, r_edge = draw_start_x + shave_edges_pixels, draw_end_x - shave_edges_pixels
                        t, b = draw_start_y + shave_edges_pixels, draw_end_y - shave_edges_pixels
                        
                        if 0 <= l < width:
                            preview[:, draw_start_y:draw_end_y, l, 2] = 1.0; preview[:, draw_start_y:draw_end_y, l, 0:2] = 0.0
                        if 0 <= r_edge < width:
                            preview[:, draw_start_y:draw_end_y, r_edge, 2] = 1.0; preview[:, draw_start_y:draw_end_y, r_edge, 0:2] = 0.0
                        if 0 <= t < height:
                            preview[:, t, draw_start_x:draw_end_x, 2] = 1.0; preview[:, t, draw_start_x:draw_end_x, 0:2] = 0.0
                        if 0 <= b < height:
                            preview[:, b, draw_start_x:draw_end_x, 2] = 1.0; preview[:, b, draw_start_x:draw_end_x, 0:2] = 0.0

        # Production matrix crop and image tensor extraction
        ratios = {
            "1:1 (Square)": 1.0, 
            "16:9 (Widescreen)": 16.0 / 9.0, 
            "3:4 (Vertical)": 3.0 / 4.0,
            "9:16 (Vertical)": 9.0 / 16.0, 
            "21:9 (Ultrawide)": 21.0 / 9.0
        }
        
        # Calculate nominal panel size for uniform output resolution
        nominal_w = int(round(width / columns))
        nominal_h = int(round(height / rows))
        
        # Subtract shaved edges to find baseline width/height
        if shave_edges_pixels > 0 and nominal_w > (shave_edges_pixels * 2) and nominal_h > (shave_edges_pixels * 2):
            base_w = nominal_w - (shave_edges_pixels * 2)
            base_h = nominal_h - (shave_edges_pixels * 2)
        else:
            base_w = nominal_w
            base_h = nominal_h
            
        base_w = max(1, base_w)
        base_h = max(1, base_h)
        
        # Define target dimensions for the batch based on ratio
        if target_aspect_ratio == "original_proportions":
            target_w = base_w
            target_h = base_h
        else:
            ratio = ratios[target_aspect_ratio]
            if ratio < 1.0:
                # Vertical format: height is the baseline to avoid upscaling
                target_h = base_h
                target_w = int(round(base_h * ratio))
            else:
                # Landscape/Square format: width is the baseline
                target_w = base_w
                target_h = int(round(base_w / ratio))
                
        target_w = max(1, target_w)
        target_h = max(1, target_h)
        
        final_slices = []

        for b in range(batch_size):
            single_img = image[b]
            for r in range(rows):
                for c in range(columns):
                    start_x = int(round(c * segment_w)) + offset_x
                    end_x = int(round((c + 1) * segment_w)) + offset_x
                    start_y = int(round(r * segment_h)) + offset_y
                    end_y = int(round((r + 1) * segment_h)) + offset_y

                    # Clamp boundaries securely to prevent canvas indexing overflow
                    start_x = max(0, min(width, start_x))
                    end_x = max(0, min(width, end_x))
                    start_y = max(0, min(height, start_y))
                    end_y = max(0, min(height, end_y))

                    if c == columns - 1: end_x = width
                    if r == rows - 1: end_y = height
                    
                    # Core 2D matrix bounding box slice extraction
                    cropped = single_img[start_y:end_y, start_x:end_x, :]
                    
                    # Inner pixel shaving execution layer
                    if shave_edges_pixels > 0 and cropped.shape[0] > (shave_edges_pixels * 2) and cropped.shape[1] > (shave_edges_pixels * 2):
                        cropped = cropped[shave_edges_pixels:-shave_edges_pixels, shave_edges_pixels:-shave_edges_pixels, :]
                    
                    curr_h, curr_w, c_ch = cropped.shape
                    if curr_h <= 0 or curr_w <= 0: continue

                    if target_aspect_ratio == "original_proportions":
                        ts_p = cropped.permute(2, 0, 1).unsqueeze(0)
                        res = torch.nn.functional.interpolate(ts_p, size=(target_h, target_w), mode='bicubic', align_corners=True)
                    else:
                        ratio = ratios[target_aspect_ratio]
                        if padding_mode == "center_crop_zoom":
                            if (curr_w / curr_h) > ratio:
                                nw = int(round(ratio * curr_h))
                                nw = max(1, min(curr_w, nw))
                                crop_start = (curr_w - nw) // 2
                                final_img = cropped[:, crop_start : crop_start + nw, :]
                            else:
                                nh = int(round(curr_w / ratio))
                                nh = max(1, min(curr_h, nh))
                                crop_start = (curr_h - nh) // 2
                                final_img = cropped[crop_start : crop_start + nh, :, :]
                                
                            ts_p = final_img.permute(2, 0, 1).unsqueeze(0)
                            res = torch.nn.functional.interpolate(ts_p, size=(target_h, target_w), mode='bicubic', align_corners=True)
                            
                            # High-definition convolution sharpening sequence (Lanczos Emulation)
                            if target_w > curr_w:
                                res = torch.nn.functional.pad(res, (1, 1, 1, 1), mode='replicate')
                                kernel = torch.tensor([[0.0, -0.1, 0.0], [-0.1, 1.4, -0.1], [0.0, -0.1, 0.0]], device=ts_p.device, dtype=ts_p.dtype)
                                res = torch.nn.functional.conv2d(res, kernel.view(1, 1, 3, 3).repeat(c_ch, 1, 1, 1), groups=c_ch)
                        else:
                            # Native letterbox padding matrix adaptation sequence
                            ts_p = cropped.permute(2, 0, 1).unsqueeze(0)
                            if (curr_w / curr_h) > ratio:
                                # Image is wider than target aspect ratio -> fit width, pad height
                                tw_pad = target_w
                                th_pad = int(round(target_w * (curr_h / curr_w)))
                                th_pad = max(1, min(target_h, th_pad))
                                res_base = torch.nn.functional.interpolate(ts_p, size=(th_pad, tw_pad), mode='bicubic', align_corners=True)
                                t_pad = target_h - th_pad
                                if t_pad > 0:
                                    res = torch.nn.functional.pad(res_base, (0, 0, t_pad // 2, t_pad - (t_pad // 2)), mode='constant', value=0.0)
                                else:
                                    res = res_base
                            else:
                                # Image is taller than target aspect ratio -> fit height, pad width
                                th_pad = target_h
                                tw_pad = int(round(target_h * (curr_w / curr_h)))
                                tw_pad = max(1, min(target_w, tw_pad))
                                res_base = torch.nn.functional.interpolate(ts_p, size=(th_pad, tw_pad), mode='bicubic', align_corners=True)
                                t_pad = target_w - tw_pad
                                if t_pad > 0:
                                    res = torch.nn.functional.pad(res_base, (t_pad // 2, t_pad - (t_pad // 2), 0, 0), mode='constant', value=0.0)
                                else:
                                    res = res_base
                                    
                            # Ensure shape is exactly (target_h, target_w)
                            if res.shape[2] != target_h or res.shape[3] != target_w:
                                res = torch.nn.functional.interpolate(res, size=(target_h, target_w), mode='bicubic', align_corners=True)

                    final_slices.append(torch.clamp(res, 0.0, 1.0).squeeze(0).permute(1, 2, 0).unsqueeze(0))

        output_tensor = torch.cat(final_slices, dim=0) if final_slices else image
        return (output_tensor, preview)
