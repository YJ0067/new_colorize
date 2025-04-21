import os
import uuid
import torch
import functools
from pathlib import Path
import torch.serialization
from flask import Flask, request, send_file, jsonify, render_template
import matplotlib
matplotlib.use('Agg')

# Allow DeOldify model loading
torch.serialization.add_safe_globals([functools.partial, getattr, slice])

# === DeOldify Imports ===
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer

# === GPU Setup ===
device.set(device=DeviceId.GPU0)
torch.backends.cudnn.benchmark = True

# === Flask Setup ===
app = Flask(__name__, template_folder='templates')

# === Model Setup ===
colorizer = get_image_colorizer(artistic=True)
colorizer.watermarked = False

# === Dummy watermark image setup ===
import numpy as np
import cv2

resource_dir = Path('./resource_images')
resource_dir.mkdir(exist_ok=True)
dummy_watermark_path = resource_dir / 'watermark.png'

if not dummy_watermark_path.exists():
    dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
    cv2.imwrite(str(dummy_watermark_path), dummy_img)

colorizer.watermark_path = str(dummy_watermark_path)

# === Folder Setup ===
UPLOAD_FOLDER = Path('uploads')
RESULT_FOLDER = Path('results')

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'source_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        render_factor = int(request.form.get('render_factor', 12))
    except ValueError:
        return jsonify({"error": "Invalid render factor"}), 400

    img_file = request.files['source_image']
    file_ext = Path(img_file.filename).suffix
    filename = f"{uuid.uuid4()}{file_ext}"
    input_path = UPLOAD_FOLDER / filename
    output_path = RESULT_FOLDER / f"{uuid.uuid4()}_result.jpg"

    img_file.save(str(input_path))

    try:
        # Get transformed image as PIL object
        result_img = colorizer.get_transformed_image(
            path=input_path,
            render_factor=render_factor,
            post_process=True
        )

        # Save result to disk manually
        result_img.save(output_path)

        return send_file(str(output_path), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if input_path.exists():
            input_path.unlink()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





