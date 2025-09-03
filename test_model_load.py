
import torch
print("Step 1: Imported torch")
from diffusers import StableDiffusionPipeline
print("Step 2: Imported StableDiffusionPipeline")

model_path = "./realisticVisionV60B1_v51HyperVAE.safetensors"
print(f"Step 3: Model path is {model_path}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Step 4: Device is {device}")

try:
    print("Step 5: Attempting to load model...")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    print("Step 6: Model loaded, moving to device...")
    pipe = pipe.to(device)
    print("Step 7: Model loaded successfully!")
except Exception as e:
    print(f"Step 8: Model loading failed: {e}")
    with open("model_load_error.log", "a", encoding="utf-8") as f:
        f.write(str(e) + "\n")
