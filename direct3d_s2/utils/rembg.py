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

    def run(self, image_arg):
        self._load_model()

        pil_image = None
        if isinstance(image_arg, Image.Image):
            pil_image = image_arg
        elif isinstance(image_arg, torch.Tensor):
            image_tensor = image_arg.cpu() # Work with CPU tensor

            if image_tensor.ndim == 4:
                if image_tensor.shape[0] == 1: # Batch size 1
                    squeezed_tensor = image_tensor.squeeze(0) # Shape [H, W, C] or [C, H, W]
                    if squeezed_tensor.ndim == 3 and squeezed_tensor.shape[2] in [3, 4]: # HWC format [H, W, C]
                        # Check if it's float (0-1) or uint8 (0-255) for HWC
                        if squeezed_tensor.dtype == torch.uint8:
                            pil_image = Image.fromarray(squeezed_tensor.numpy(), 'RGB' if squeezed_tensor.shape[2] == 3 else 'RGBA')
                        else: # Assuming float tensor is CHW format for ToPILImage, so permute HWC to CHW
                            chw_tensor = squeezed_tensor.permute(2, 0, 1) # Convert HWC to CHW [C, H, W]
                            pil_image = transforms.ToPILImage()(chw_tensor)
                    elif squeezed_tensor.ndim == 3 and squeezed_tensor.shape[0] in [3, 4]: # CHW format already [C, H, W]
                        pil_image = transforms.ToPILImage()(squeezed_tensor)
                    else:
                        raise ValueError(f"Received a 4D Tensor (squeezed to 3D) in BiRefNet.run with unhandled shape: {squeezed_tensor.shape}")
                else: # Batch size > 1
                    raise ValueError(f"Received a 4D Tensor with batch size > 1 in BiRefNet.run, which is not supported: {image_tensor.shape}")
            elif image_tensor.ndim == 3: # Handling for 3D tensors
                if image_tensor.shape[0] in [3, 4]: # CHW format [C, H, W] (e.g., float 0-1 or uint8)
                    pil_image = transforms.ToPILImage()(image_tensor)
                elif image_tensor.shape[2] in [3, 4] and image_tensor.dtype == torch.uint8: # HWC format [H, W, C], uint8
                    if image_tensor.shape[2] == 4: # HWCA
                        pil_image = Image.fromarray(image_tensor.numpy(), 'RGBA')
                    else: # HWC (RGB)
                        pil_image = Image.fromarray(image_tensor.numpy(), 'RGB')
                else: # Other 3D shapes/dtypes
                    raise ValueError(f"Received a 3D Tensor in BiRefNet.run of unhandled shape/dtype: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            elif image_tensor.ndim == 2: # Grayscale (H, W)
                pil_image = transforms.ToPILImage()(image_tensor.unsqueeze(0)) # Add channel dim for ToPILImage
            else: # Other dimensions (e.g., 1D, >4D)
                raise ValueError(f"Received a Tensor in BiRefNet.run with unhandled dimensions: {image_tensor.ndim}, shape: {image_tensor.shape}")
        else: # Not PIL.Image and not torch.Tensor
            raise TypeError(f"BiRefNet.run received an unexpected image type: {type(image_arg)}")

        if pil_image is None:
             raise RuntimeError("pil_image was not set, unexpected state in BiRefNet.run")

        if pil_image.mode == 'RGBA' or pil_image.mode == 'P' or pil_image.mode == 'L' or pil_image.mode == 'CMYK' or pil_image.mode == 'YCbCr':
             image_rgb = pil_image.convert('RGB')
        elif pil_image.mode == 'RGB':
             image_rgb = pil_image
        else:
             try:
                 image_rgb = pil_image.convert('RGB')
             except Exception as e:
                 raise ValueError(f"Could not convert PIL image mode '{pil_image.mode}' to RGB in BiRefNet.run. Error: {e}")


        image_size = (1024, 1024)

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
