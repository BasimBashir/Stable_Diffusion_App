from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, load_image_dino, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import torch
import numpy as np
import glob
import os

SD_Inpainting_model = "StableDiffusion models/SD_Inpainting"
sam_model = "SAM_and_GroundingDino_models_weights/sam_vit_h_4b8939.pth"
grounding_dino_model = "SAM_and_GroundingDino_models_weights/groundingdino_swint_ogc.pth"
grounding_dino_model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# -----Set Image and CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------SAM Parameters
model_type = "vit_h"
predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_model).to(device=device))

# ------Stable Diffusion
pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_Inpainting_model, torch_dtype=torch.float16,).to(device)
# pipe.save_pretrained(SD_Inpainting_model)

# ----Grounding DINO
groundingdino_model = load_model(grounding_dino_model_config, grounding_dino_model)


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.9])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))



def process_boxes(boxes, src):
    H, W, _ = src.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    return predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)


def Inpaint_Image(prompt, CLASS, box_threshold, text_threshold, negative_prompt=None, num_inference_steps: int = 20):
    '''
    - path (str): Path to the image file
    - item (str): Item to be recognized in the image
    - prompt (str): Item to replace the selected object in the image
    - box_threshold (float): Threshold for the bounding box predictions.
    - text_threshold (float): Threshold for the text predictions.
    '''

    # coco_classes_list = "person . bicycle . car . motorcycle . airplane . bus . eiffel tower .train . truck . boat. traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"

    # path = "static/user_uploads/user.jpg"

    image_dir = "static/user_uploads/"

    image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
        os.path.join(image_dir, "*.jpeg"))

    src, img = load_image_dino(image_path[0])

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=CLASS,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    predictor.set_image(src)
    new_boxes = process_boxes(boxes, src)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=new_boxes,
        multimask_output=False,
    )

    img_annotated_mask = show_mask(masks[0][0].cpu(),
        annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
    )

    if negative_prompt is not None:
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, image=Image.fromarray(src).resize((512, 512)), mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))).images[0]
    
    else:
        image = pipe(prompt=prompt, 
                    negative_prompt="bad anatomy, bad proportions, blurry, cloned face, cropped, deformed, dehydrated, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, too many fingers, ugly, username, watermark, worst quality",
                    num_inference_steps=num_inference_steps, 
                    image=Image.fromarray(src).resize((512, 512)), 
                    mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))).images[0]
    
    return image 
