from diffusers import DiffusionPipeline
import torch
import os


base_model_weights = "StableDiffusion models\SDXL_Base_model"
refiner_model_weights = "StableDiffusion models\SDXL_Refiner_model"

if not (os.path.exists(base_model_weights) and os.path.exists(refiner_model_weights)):

    os.makedirs(base_model_weights)
    os.makedirs(refiner_model_weights)

    print(f"The folder '{base_model_weights}' and '{refiner_model_weights}' has been created.")

    # load both base & refiner
    base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=base.text_encoder_2, vae=base.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    refiner.to("cuda")

    base.save_pretrained(base_model_weights)
    refiner.save_pretrained(refiner_model_weights)
else:
    print(f"The folder '{base_model_weights}' and '{refiner_model_weights}' already exists.")
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        base_model_weights,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True)

    base.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        refiner_model_weights,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")

    refiner.to("cuda")


def generate_text2image_quality(prompt, negative_prompt=None, inference_steps: int = 40, high_noise_frac: float = 0.8):
    # Define how many steps and what % of steps to be run on each expert (80/20) here
    # n_steps = 40
    # high_noise_frac = 0.8

    if negative_prompt is not None:
        # run both experts
        image = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=inference_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
    else:
        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=inference_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

    return image
