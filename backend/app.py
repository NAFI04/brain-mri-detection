# app.py
import os
import json
import base64
import time
from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from gradcam import make_gradcam_heatmap, overlay_heatmap

app = Flask(__name__, template_folder='templates')

# Load model and labels
MODEL_PATH = 'brain_mobilenet.h5'
LABEL_MAP = 'label_map.json'

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP) as f:
    index_to_label = json.load(f)

# Find last Conv2D layer for Grad-CAM
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name
        except Exception:
            continue
    return None

LAST_CONV = find_last_conv_layer(model)
print("Using Conv Layer:", LAST_CONV)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'photo' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['photo']
        ts = str(int(time.time() * 1000))
        tmp_path = f'static/overlays/tmp_{ts}.jpg'
        file.save(tmp_path)

        # Preprocess image
        #img = image.load_img(tmp_path, target_size=(224, 224))
        img = image.load_img(tmp_path, target_size=(64, 64))

        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Predict probability
        raw_pred = model.predict(img_arr)[0][0]
        # Convert raw output to probability using sigmoid
        pred_prob = 1 / (1 + np.exp(-raw_pred))  # always 0-1


        # ✅ Ensure probability is between 0 and 1
        if raw_pred > 1 or raw_pred < 0:
            pred_prob = 1 / (1 + np.exp(-raw_pred))  # sigmoid conversion
        else:
            pred_prob = raw_pred
        pred_prob = float(np.clip(pred_prob, 0.0, 1.0))  # clamp 0-1

        # Apply custom threshold (0.4)
        threshold = 0.4
        result = "Tumor Detected" if pred_prob >= threshold else "No Tumor"

        # Generate Grad-CAM overlay
        heatmap = make_gradcam_heatmap(img_arr, model, LAST_CONV)
        overlay_path = f'static/overlays/overlay_{ts}.jpg'
        overlay_heatmap(tmp_path, heatmap, overlay_path)

        # Convert overlay image to base64 for display
        with open(overlay_path, 'rb') as f:
            overlay_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'prediction': result,
            'probability': round(pred_prob * 100, 2),  # percent 0-100%
            'overlay': "data:image/jpeg;base64," + overlay_b64
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    os.makedirs('static/overlays', exist_ok=True)
    threading.Timer(1.5, open_browser).start()  # wait 1.5 sec, then open browser
    app.run(debug=False, host='127.0.0.1', port=5000)