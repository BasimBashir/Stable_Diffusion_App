import torch
from diffusers import AutoPipelineForText2Image
import random
import os


model_weights = "StableDiffusion models/SDXL_weights"

if not os.path.exists(model_weights):
    os.makedirs(model_weights)
    print(f"The folder '{model_weights}' has been created.")

    image_generator = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", force_download=True, resume_download=False
    ).to("cuda")
    image_generator.save_pretrained(model_weights)
else:
    print(f"The folder '{model_weights}' already exists.")
    image_generator = AutoPipelineForText2Image.from_pretrained(model_weights, torch_dtype=torch.float16, variant="fp16").to("cuda")


def generate_text2image_fast(prompt, negative_prompt=None, inference_steps: int = 2, guidance_scale: float = 0.0):
    seed = random.randint(1, 9999)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if negative_prompt is not None:
        outputs = image_generator(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    else:
        outputs = image_generator(
            prompt=prompt,
            negative_prompt="bad anatomy, bad proportions, blurry, cloned face, cropped, deformed, dehydrated, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, too many fingers, ugly, username, watermark, worst quality",           
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    return outputs.images[0]


# if __name__ == "__main__":
#     prompt = """
#     A vividly detailed digital artwork depicting a warrior in dramatic battle stance,
#     clad in unique armor and wielding a distinctive weapon,
#     set against an intense backdrop.
#     dramatic lighting, cinematic, post-production
#     """.strip()
#
#     generate_image(prompt, inference_steps=2)
