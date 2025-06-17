import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet(object):
    def __init__(self, device, model_path=None):
        self.device = device
        self.model_path = model_path
        self.birefnet_model = None

    def _load_model(self):
        if self.birefnet_model is not None:
            return
        if not self.model_path:
            raise ValueError("BiRefNet model path not provided or empty.")

        from transformers import AutoModelForImageSegmentation
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        ).to(self.device)
        self.birefnet_model.eval()

    def run(self, image_arg): # Renamed input to image_arg to avoid confusion
        self._load_model()

        pil_image = None
        if isinstance(image_arg, Image.Image):
            pil_image = image_arg
        elif isinstance(image_arg, torch.Tensor):
            image_tensor = image_arg.cpu()
            if image_tensor.ndim == 3 and image_tensor.shape[0] == 3: # CHW (e.g., float 0-1)
                pil_image = transforms.ToPILImage()(image_tensor)
            elif image_tensor.ndim == 3 and image_tensor.shape[0] == 4: # CHWA?
                pil_image = transforms.ToPILImage()(image_tensor[:3,:,:]) # Take RGB from RGBA
            elif image_tensor.ndim == 3 and image_tensor.shape[2] == 3 and image_tensor.dtype == torch.uint8: # HWC, uint8
                pil_image = Image.fromarray(image_tensor.numpy())
            elif image_tensor.ndim == 3 and image_tensor.shape[2] == 4 and image_tensor.dtype == torch.uint8: # HWCA, uint8
                pil_image_rgba = Image.fromarray(image_tensor.numpy(), 'RGBA')
                pil_image = pil_image_rgba.convert('RGB') # Explicitly convert here to handle alpha
            elif image_tensor.ndim == 2: # Grayscale (H, W)
                # ToPILImage expects CHW, so add channel dimension
                pil_image = transforms.ToPILImage()(image_tensor.unsqueeze(0))
            else:
                raise ValueError(f"Received a Tensor in BiRefNet.run of unhandled shape/dtype: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        else:
            raise TypeError(f"BiRefNet.run received an unexpected image type: {type(image_arg)}")

        if pil_image.mode == 'RGBA' or pil_image.mode == 'P': # P for paletted images
             image_rgb = pil_image.convert('RGB')
        elif pil_image.mode == 'L': # Grayscale
             image_rgb = pil_image.convert('RGB') # Convert grayscale to RGB
        else:
             image_rgb = pil_image # Assume it's already RGB or a mode transformable

        image_size = (1024, 1024) # Original line

        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images_tensor = transform_image(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet_model(input_images_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        mask = pred_pil.resize(image_rgb.size)
        mask_np = np.array(mask)

        image_rgb_np = np.array(image_rgb)
        final_image_np = np.concatenate([image_rgb_np, mask_np[..., None]], axis=-1)

        return final_image_np
