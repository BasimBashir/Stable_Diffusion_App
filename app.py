from flask import Flask, render_template, request, make_response, send_file, jsonify
from groundingdino.util.inference import load_model, load_image_dino, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
from diffusers.utils import load_image
from transformers import pipeline
import urllib.request
from PIL import Image
from tqdm import tqdm
import numpy as np
import warnings
import torch
import glob
import cv2
import os


# To suppress all warnings globally
warnings.filterwarnings("ignore")

app = Flask(__name__)


option = None
phrases = None


sam_model = "SAM_and_GroundingDino_models_weights/sam_vit_h_4b8939.pth"
grounding_dino_model = "SAM_and_GroundingDino_models_weights/groundingdino_swint_ogc.pth"
grounding_dino_model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# ----------------------------------------------------------------------------------------------------------------------------------- #

# URL of the file to download
SAM_model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# URL of the file to download
GroundingDino_model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# folder1 to save the SAM model
folder_path1 = "SAM_and_GroundingDino_models_weights"


if not (os.path.exists(folder_path1)):

    print(f"\n\nSAM and GroundingDino weights are not present. Dowloading weights in background at:\n{folder_path1}\n\n")

    # Create the folder1 if it doesn't exist
    os.makedirs(folder_path1, exist_ok=True)

   # Define the file paths
    file_paths = [
        os.path.join(folder_path1, "sam_vit_h_4b8939.pth"),
        os.path.join(folder_path1, "groundingdino_swint_ogc.pth"),
    ]

    # Define the model URLs
    model_urls = [
        SAM_model_url,
        GroundingDino_model_url,
    ]

    # Iterate over the file paths and model URLs
    for file_path, model_url in zip(file_paths, model_urls):
        # Download the weights with a progress bar
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(file_path)) as t:
            urllib.request.urlretrieve(model_url, file_path, reporthook=lambda block_num, block_size, total_size: t.update(block_size))

    print("\n\n************************************ Download completed! ************************************\n\n")

    # -----Set Image and CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------SAM Parameters
    model_type = "vit_h"
    predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_model).to(device=device))

    print(f"\n\nSAM and GroundingDino weights are downloaded and Loaded from:\n{folder_path1}\n\n")

else:

    print(f"\n\nSAM and GroundingDino weights present. Loading weights from:\n{folder_path1}\n\n")

    # -----Set Image and CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------SAM Parameters
    model_type = "vit_h"
    predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_model).to(device=device))

    # ----Grounding DINO
    groundingdino_model = load_model(grounding_dino_model_config, grounding_dino_model)

# ----------------------------------------------------------------------------------------------------------------------------------- #


@app.route('/')
def text2image():
    response = make_response(render_template('text2img.html'))
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response


@app.route('/generate_text2img', methods=['POST'])
def generate():
    data = request.get_json()
    print(f"Prompt: {data['prompt']}")
    print(f"Negative Prompt: {data['negative_prompt']}")
    print(f"Higher_Quality: {data['toggleState']}")
    print(f"Inference Steps: {int(data['inferenceSteps'])}")

    prompt = f"""
    {data['prompt']}
    """.strip()

    negative_prompt = f"""
    {data['negative_prompt']}
    """.strip()

    if data['toggleState'] == "off":
        from Run_text2Img_Inference_Fast import generate_text2image_fast
        # generate image faster but with less quality
        image = generate_text2image_fast(prompt, negative_prompt=negative_prompt, inference_steps=int(data['inferenceSteps']))
        image.save("static/images/generated_image.png")
        # print("Image sending to server.")
    else:
        from Run_text2Img_inference_Quality import generate_text2image_quality
        # generate image slower but with greater quality
        image = generate_text2image_quality(prompt, negative_prompt=negative_prompt, inference_steps=int(data['inferenceSteps']))
        image.save("static/images/generated_image.png")
        # print("Image sending to server.")

    return {"status": "success"}


@app.route('/download_text2image')
def download():
    return send_file('static/images/generated_image.png', mimetype='image/png', as_attachment=True)


@app.route('/img2img')
def img2img():

    try:
        os.remove("static/ControlNet_Image/controlImage.png")
    except Exception:
        pass
    
    try:
        os.remove("static/user_uploads/user.png")
    except Exception:
        pass
    
    try:
        os.remove("static/user_uploads/user.jpg")
    except Exception:
        pass
    
    try:
        os.remove("static/user_uploads/user.jpeg")
    except Exception:
        pass

    try:
        os.remove("static/images/generated_image.png")
    except Exception:
        pass
    
    print(f"Images deleted successfully.")
        
    response = make_response(render_template('img2img.html'))
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response


@app.route('/upload', methods=['POST'])
def upload_image():

    global option
    global phrases

    # Save image_canny as "controlImage.png"
    control_image_path = "static/ControlNet_Image/controlImage.png"

    # depth Estimator model
    detailed_depth_estimator_model = "StableDiffusion models/Detailed_Depth_Estimator_Model"

    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            option = request.form['option']

            # Rename the image to "user" before saving
            image.filename = 'user' + os.path.splitext(image.filename)[1]

            # Save the renamed image to the 'static/user_uploads' directory
            image.save(os.path.join('static/user_uploads', image.filename))
            print('Image saved!')

            image_dir = "static/user_uploads/"

            image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
                os.path.join(image_dir, "*.jpeg"))

            init_image = load_image(image_path[0])

            # Preserve the aspect ratio while resizing to fit within 512x512
            init_image.thumbnail((512, 512))

            image_input = np.array(init_image)

            if option == "Canny":
                print("User selected Canny")

                # define parameters from canny edge detection
                low_threshold = 100
                high_threshold = 200 

                # do canny edge detection
                image_canny = cv2.Canny(image_input, low_threshold, high_threshold)

                # convert to PIL image format
                image_canny = image_canny[:, :, None]
                image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
                image_canny = Image.fromarray(image_canny)

                image_canny.save(control_image_path)
            
            elif option == "Depth":
                print("User selected Depth")
                depth_estimator = pipeline("depth-estimation", model=detailed_depth_estimator_model)

                image_input = Image.fromarray(image_input)

                # do all the preprocessing to get the normal image
                image = depth_estimator(image_input)['predicted_depth'][0]

                image = image.numpy()

                image_depth = image.copy()
                image_depth -= np.min(image_depth)
                image_depth /= np.max(image_depth)

                bg_threhold = 0.4

                x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
                x[image_depth < bg_threhold] = 0

                y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
                y[image_depth < bg_threhold] = 0

                z = np.ones_like(x) * np.pi * 2.0

                image = np.stack([x, y, z], axis=2)
                image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
                image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                image_normal = Image.fromarray(image)

                image_normal.save(control_image_path)
            
            elif option == "Segmentation":
                print("User selected Segmentation")

                from SD_SAM import process_boxes, show_mask

                # image_path = "static/user_uploads/user.jpg"

                image_dir = "static/user_uploads/"

                image_path = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
                    os.path.join(image_dir, "*.jpeg"))

                box_threshold = 0.3
                text_threshold = 0.25

                coco_classes_list = "face . eyes . ears . lips . nails . fingers . legs . feet . nose . eyebrows . moustache . hair . reflective safety vest . helmet . head . nonreflective safety vest . mens shadow . womens shadow . shadow . straw . chair . dog . table . shoe . light bulb . coffee . hat . glass . car . tail . umbrella . desk . wall . shirt . pants . hoodie . jacket . dress . ground . background . chair . carpet . building . flower . mirror . couch . sofa . cushion . sky . tree . person . bicycle . car . motorcycle . airplane . bus . train . truck . boat. traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"
                
                src, img = load_image_dino(image_path[0])


                boxes, logits, phrases = predict(
                    model=groundingdino_model,
                    image=img,
                    caption=coco_classes_list,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )

                print("Total Detected Class/Classes: ", phrases)
                

                predictor.set_image(src)
                new_boxes = process_boxes(boxes, src)

                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=new_boxes,
                    multimask_output=False,
                )

                # Generate annotated mask image
                img_annotated_mask = show_mask(masks[0][0].cpu(),
                    annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
                )

                # Convert the annotated mask to a PIL image
                annotated_mask_pil = Image.fromarray(img_annotated_mask)

                # Save the PIL image to the specified path
                annotated_mask_pil.save(control_image_path)
  
            else:
                print("something went wrong in user's checkbox selection Logic")
                pass

            return jsonify({'message': 'Success'})

    return jsonify({'error': 'Missing image file or incorrect API call'})


@app.route('/get_phrases')
def get_phrases():
    global phrases
    return jsonify({'phrases': phrases})


@app.route('/generate_img2img', methods=['POST'])
def generate2():

    data = request.get_json()
    print(f"Prompt: {data['prompt']}")
    print(f"Strength: {int(data['strength'])}")
    print(f"Selected Class: {data['phrases']}")

    prompt = f"""
    {data['prompt']}
    """.strip()

    if option == "Canny":
        from Run_Img2Img_Inference_Canny import generate_image2image_canny
        
        # generate from canny image
        image = generate_image2image_canny(prompt=prompt, num_inference_steps=int(data['strength']))
        image.save("static/images/generated_image.png")

    elif option == "Depth":
        from Run_Img2Img_Inference_Depth import generate_image2image_depth

        # generate from depth image
        image = generate_image2image_depth(prompt=prompt, num_inference_steps=int(data['strength']))
        image.save("static/images/generated_image.png")

    elif option == "Segmentation":
        from SD_SAM import Inpaint_Image

        # generate from segmentated image
        image = Inpaint_Image(prompt=prompt, CLASS=data['phrases'][0], box_threshold=0.5, text_threshold=0.2, num_inference_steps=int(data['strength']))
        image.save("static/images/generated_image.png")
        

    else:
        print("something went wrong in user's checkbox selection Logic")
        pass

    return {"status": "success"}


@app.route('/download_image2image')
def download2():
    return send_file('static/images/generated_image.png',
                     mimetype='image/png',
                     as_attachment=True
                     )


@app.route('/text2video')
def text2video():
    return "This Feature is Coming Soon!"


if __name__ == '__main__':
    app.run(host='0.0.0.0')

    # ngrok http --domain=glorious-allegedly-weasel.ngrok-free.app 5000
