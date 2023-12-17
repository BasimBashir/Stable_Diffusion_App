import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image
import glob
import os

model_weights = "StableDiffusion models\SD_img2img"

if not os.path.exists(model_weights):
    os.makedirs(model_weights)
    print(f"The folder '{model_weights}' has been created.")

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.save_pretrained(model_weights)

else:
    print(f"The folder '{model_weights}' already exists.")
    # Load Model
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
       model_weights,
       torch_dtype=torch.float16,
    ).to("cuda")


def generate_image2image(prompt, negative_prompt=None, strength: float = 0.8):

    image_dir = "static/user_uploads/"
    image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
        os.path.join(image_dir, "*.jpeg"))

    init_image = load_image(image_path[0])  # load_image("static/user_uploads/user.jpg")
    if negative_prompt is not None:
        image = pipe(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=strength).images[0]
    else:
        image = pipe(prompt=prompt, image=init_image, strength=strength).images[0]

    return image
