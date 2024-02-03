import torch
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import glob
import os


detailed_depth_controlNet = "StableDiffusion models/Detailed_Depth_ControlNet"
detailed_depth_estimator_model = "StableDiffusion models/Detailed_Depth_Estimator_Model"
SD_model = "StableDiffusion models/ControlNet_Compatible_SD_model"

if not (os.path.exists(detailed_depth_controlNet) and os.path.exists(detailed_depth_estimator_model) and os.path.exists(SD_model)):

    os.makedirs(detailed_depth_controlNet)
    os.makedirs(detailed_depth_estimator_model)
    os.makedirs(SD_model)

    print(f"The folders '{detailed_depth_controlNet}' and '{detailed_depth_estimator_model}' and '{SD_model}' has been created.")

    # load the Dense Prediction Transformer (DPT) model for getting normal maps
    depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas")
    depth_estimator.save_pretrained(detailed_depth_estimator_model)

    # load the controlnet model for normal maps
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
    )
    controlnet.save_pretrained(detailed_depth_controlNet)

    # load the stable diffusion pipeline with controlnet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained(SD_model)

else:
    print(f"The folders '{detailed_depth_controlNet}' and '{detailed_depth_estimator_model}' and '{SD_model}' already exists.")
    
    # load the Dense Prediction Transformer (DPT) model for getting normal maps
    # depth_estimator = pipeline("depth-estimation", model=detailed_depth_estimator_model)

    # load the controlnet model for normal maps
    controlnet = ControlNetModel.from_pretrained(
        detailed_depth_controlNet, torch_dtype=torch.float16
    )

    # load the stable diffusion pipeline with controlnet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # enable efficient implementations using xformers for faster inference
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()


def generate_image2image_depth(prompt, negative_prompt=None, num_inference_steps: int = 20):

    image_dir = "static/ControlNet_Image"
    image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
        os.path.join(image_dir, "*.jpeg"))

    init_image = load_image(image_path[0])

    if negative_prompt is not None:
        image = pipe(prompt=prompt, image=init_image, num_inference_steps=num_inference_steps).images[0]

    else:
        image = pipe(prompt=prompt, image=init_image, negative_prompt="ugly, bad quality, animated, painting, blur, cartoonish, hand painting, bad shape, disfigured, low resolution, ugly", num_inference_steps=num_inference_steps).images[0]


    return image
