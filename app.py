from flask import Flask, render_template, request, make_response, send_file, jsonify
from PIL import Image
import os

app = Flask(__name__)


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
    return send_file('static/images/generated_image.png',
                     mimetype='image/png',
                     as_attachment=True
                     )


@app.route('/img2img')
def img2img():
    response = make_response(render_template('img2img.html'))
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response


@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']

            # Rename the image to "user" before saving
            image.filename = 'user' + os.path.splitext(image.filename)[1]

            # Save the renamed image to the 'static/user_uploads' directory
            image.save(os.path.join('static/user_uploads', image.filename))
            print('Image saved!')
            return jsonify({'message': 'Image uploaded successfully!'})

    return jsonify({'error': 'Missing image file or incorrect API call'})


@app.route('/generate_img2img', methods=['POST'])
def generate2():
    from Run_Img2Img_Inference import generate_image2image
    data = request.get_json()
    print(f"Prompt: {data['prompt']}")
    print(f"Strength: {float(data['strength'])}")

    prompt = f"""
    {data['prompt']}
    """.strip()

    # generate image
    image = generate_image2image(prompt=prompt, strength=float(data['strength']))
    image.save("static/images/generated_image.png")
    # print("Image sending to server.")

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
    app.run(debug=True)
