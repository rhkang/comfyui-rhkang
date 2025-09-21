import torch
from comfy.model_management import get_torch_device

_presets = [
    ("1024x1024 (square)", (1024, 1024)),
    ("1152x896 (landscape)", (1152, 896)),
    ("896x1152 (portrait)", (896, 1152)),
    ("1216x832 (landscape)", (1216, 832)),
    ("832x1216 (portrait)", (832, 1216)),
    ("1344x768 (landscape)", (1344, 768)),
    ("768x1344 (portrait)", (768, 1344)),
    ("1536x640 (landscape)", (1536, 640)),
    ("640x1536 (portrait)", (640, 1536)),
]

class SDXLEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        presets = _presets
        return {
        "required": {
            "resolution": ([p[0] for p in presets],),
            "random": ("BOOLEAN", {"default": False}),
            "mode": (["all", "landscape", "portrait"], {"default": "all", "tooltip": "Resolution candidates will be filtered by this mode. Ignored if random is False."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
        }
    }

    PRESET_MAP = dict(_presets)

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "rhkang/latent"

    @classmethod
    def IS_CHANGED(cls, resolution, random, mode, batch_size):
        if random:
            return float("NaN")

        return False

    def generate(self, resolution, random, mode, batch_size):
        preset_keys = list(self.PRESET_MAP.keys())
        
        if random:
            if mode == "landscape":
                filtered_keys = [key for key in preset_keys if "landscape" in key]
            elif mode == "portrait":
                filtered_keys = [key for key in preset_keys if "portrait" in key]
            else:
                filtered_keys = preset_keys
            
            idx = torch.randint(0, len(filtered_keys), (1,)).item()
            resolution = filtered_keys[idx]
            
        width, height = self.PRESET_MAP[resolution]
        device = get_torch_device()
        channels = 4
        latent = torch.zeros((batch_size, channels, height // 8, width // 8), device=device)
        return ({"samples": latent},)

NODE_CLASS_MAPPINGS = {
    "SDXLEmptyLatentImage": SDXLEmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLEmptyLatentImage": "SDXL Empty Latent Image",
}