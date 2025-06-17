import numpy as np  
import torch
from torchvision import transforms


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

    def run(self, image):
        self._load_model()
        image = image.convert('RGB')
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask = np.array(mask)
        image = np.concatenate([np.array(image), mask[..., None]], axis=-1)
        return image
