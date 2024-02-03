import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import glob
import os


canny_controlNet = "StableDiffusion models/Canny_ControlNet"
SD_model = "StableDiffusion models/ControlNet_Compatible_SD_model"

if not (os.path.exists(canny_controlNet) and os.path.exists(SD_model)):
    os.makedirs(canny_controlNet)
    os.makedirs(SD_model)
    print(f"The folders '{canny_controlNet}' and '{SD_model}' has been created.")

    # load the controlnet model for canny edge detection
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )

    controlnet.save_pretrained(canny_controlNet)

    # load the stable diffusion pipeline with controlnet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.save_pretrained(SD_model)

else:
    print(f"The folders '{canny_controlNet}' and '{SD_model}' already exists.")
    
    # load the controlnet model for canny edge detection
    controlnet = ControlNetModel.from_pretrained(
        canny_controlNet, torch_dtype=torch.float16
    )

    # load the stable diffusion pipeline with controlnet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # enable efficient implementations using xformers for faster inference
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()


def generate_image2image_canny(prompt, negative_prompt=None, num_inference_steps: int = 20):

    image_dir = "static/ControlNet_Image"
    image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
        os.path.join(image_dir, "*.jpeg"))

    init_image = load_image(image_path[0])

    if negative_prompt is not None:
        image = pipe(prompt=prompt, image=init_image, num_inference_steps=num_inference_steps).images[0]
    else:
        image = pipe(prompt=prompt, image=init_image, negative_prompt="ugly, bad quality, animated, painting, blur, cartoonish, hand painting, bad shape, disfigured, low resolution, ugly", num_inference_steps=num_inference_steps).images[0]


    return image
