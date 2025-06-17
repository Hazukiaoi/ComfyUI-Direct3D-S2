from direct3d_s2.pipeline import Direct3DS2Pipeline
import folder_paths
import os


class LoadDirect3DS2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "wushuang98/Direct3D-S2"}),
                "subfolder_path": ("STRING", {"default": "direct3d-s2-v-1-1"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "birefnet_model_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Direct3D‑S2"

    def load_model(self, model_path, subfolder_path, device, birefnet_model_path):
        pipeline = Direct3DS2Pipeline.from_pretrained(
          model_path, 
          subfolder=subfolder_path,
          birefnet_model_path=birefnet_model_path
        )
        pipeline.to(device)
        model = pipeline
        
        return (model,)


class LoadDirect3DS2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "assets/test/13.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "input_image"
    CATEGORY = "Direct3D‑S2"

    def input_image(self, image_path):
        image = image_path
        return (image,)


class Direct3DS2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "sdf_resolution": ("INT", {"default": 1024}),
                "remesh": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"
    CATEGORY = "Direct3D‑S2"

    def generate(self, model, image, sdf_resolution, remesh):
        pipeline = model
        
        mesh = pipeline(
          image, 
          sdf_resolution=sdf_resolution, # 512 or 1024
          remesh=remesh, # Switch to True if you need to reduce the number of triangles.
        )["mesh"]
        
        return (mesh,)


class SaveDirect3DS2Mesh:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "Direct3D_S2_mesh.obj"}),
                "mesh": ("MESH",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "Direct3D‑S2"

    def save(self, filename, mesh):
        output_dir = folder_paths.get_output_directory()
        base_name, extension = os.path.splitext(filename)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        counter = 1
        current_filename_to_check = filename
        full_path = os.path.join(output_dir, current_filename_to_check)

        actual_filename_for_saving = filename
        while os.path.exists(full_path):
            actual_filename_for_saving = f"{base_name}_{counter:03d}{extension}"
            full_path = os.path.join(output_dir, actual_filename_for_saving)
            counter += 1

        mesh.export(full_path)
        
        return { "ui": { "text": [f"Saved mesh as {actual_filename_for_saving} in ComfyUI output directory."] } }


