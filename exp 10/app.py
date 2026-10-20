"""
Blood Cell Image Classification Using Machine Learning
======================================================
Flask Web Application for real-time blood cell image classification
using a trained Convolutional Neural Network (CNN).
"""

import os
import json
import time
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

import tensorflow as tf
from tensorflow import keras

# Application Configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = 'blood-cell-classification-ml-lab-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'blood_cell_cnn.keras')
METRICS_PATH = os.path.join(BASE_DIR, 'static', 'plots', 'metrics.json')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = (128, 128)

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'plots'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'sample_images'), exist_ok=True)

# Blood Cell Clinical Knowledge Base
CELL_INFO = {
    'Eosinophil': {
        'name': 'Eosinophil',
        'type': 'Granulocyte',
        'morphology': 'Prominent bilobed nucleus with large, coarse reddish-orange cytoplasmic granules.',
        'normal_range': '1% - 4% of total WBC count (50 - 400 cells/µL)',
        'clinical_function': 'Plays a crucial role in defending against parasitic worm infections and mediates allergic and asthma reactions.',
        'badge_color': '#f97316'  # Orange
    },
    'Lymphocyte': {
        'name': 'Lymphocyte',
        'type': 'Agranulocyte',
        'morphology': 'Large, dense, spherical nucleus with minimal rim of pale blue cytoplasm.',
        'normal_range': '20% - 40% of total WBC count (1,000 - 4,000 cells/µL)',
        'clinical_function': 'Key effector of adaptive immunity; includes B-cells (antibody production), T-cells (cell-mediated immunity), and Natural Killer (NK) cells.',
        'badge_color': '#3b82f6'  # Blue
    },
    'Monocyte': {
        'name': 'Monocyte',
        'type': 'Agranulocyte',
        'morphology': 'Largest leukocyte with an indented kidney- or horseshoe-shaped nucleus and abundant grayish-blue cytoplasm.',
        'normal_range': '2% - 8% of total WBC count (100 - 800 cells/µL)',
        'clinical_function': 'Circulates in blood before migrating into tissues to differentiate into macrophages and dendritic cells for phagocytosis and antigen presentation.',
        'badge_color': '#14b8a6'  # Teal
    },
    'Neutrophil': {
        'name': 'Neutrophil',
        'type': 'Granulocyte',
        'morphology': 'Multilobed nucleus (3 to 5 lobes connected by thin chromatic strands) with fine, neutral pink/lilac granules.',
        'normal_range': '50% - 70% of total WBC count (2,500 - 7,000 cells/µL)',
        'clinical_function': 'Primary primary first-responder phagocyte that rapidly migrates to sites of acute bacterial infection and inflammation.',
        'badge_color': '#ef4444'  # Red
    }
}

# Global Model & Classes Cache
_model = None
_classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

def get_model():
    """Load and cache the trained Keras CNN model."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading trained CNN model from '{MODEL_PATH}'...")
            try:
                _model = keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                _model = None
        else:
            print(f"Model file not found at '{MODEL_PATH}'. Please train the model first.")
            _model = None
    return _model

def get_classes():
    """Dynamically determine classes from dataset directory or defaults."""
    global _classes
    if os.path.exists(DATASET_DIR):
        detected = [
            d for d in sorted(os.listdir(DATASET_DIR))
            if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')
        ]
        if detected:
            _classes = detected
    return _classes

def allowed_file(filename):
    """Validate file extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_file(image_path, img_size=IMG_SIZE):
    """
    Load an image, convert to RGB, resize to 128x128, and normalize to [0, 1].
    Returns preprocessed numpy array ready for CNN prediction.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        orig_dimensions = (img_rgb.width, img_rgb.height)
        img_resized = img_rgb.resize(img_size, Image.Resampling.BILINEAR)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        # Expand dimensions to (1, 128, 128, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        return input_tensor, orig_dimensions

def load_metrics():
    """Load evaluation metrics JSON file if generated."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading metrics JSON: {e}")
    return None

def get_sample_images():
    """Collect available sample demo images for quick 1-click testing."""
    samples = {}
    samples_dir = os.path.join(BASE_DIR, 'static', 'sample_images')
    if os.path.exists(samples_dir):
        for c in get_classes():
            c_dir = os.path.join(samples_dir, c)
            if os.path.exists(c_dir):
                files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    samples[c] = [f"sample_images/{c}/{f}" for f in sorted(files)]
    return samples

@app.route('/')
def index():
    """Main application landing page."""
    model = get_model()
    classes = get_classes()
    metrics = load_metrics()
    sample_images = get_sample_images()
    model_loaded = model is not None

    return render_template(
        'index.html',
        model_loaded=model_loaded,
        classes=classes,
        metrics=metrics,
        cell_info=CELL_INFO,
        sample_images=sample_images
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle blood cell image classification:
    - Accepts file upload via 'file' multipart form field
    - OR accepts sample image path via 'sample_path' form field
    - Runs image preprocessing (RGB, 128x128, [0, 1] normalization)
    - Feeds input tensor to CNN model
    - Returns predicted class, confidence, all probabilities, and cell details
    """
    model = get_model()
    if model is None:
        return jsonify({
            'success': False,
            'error': 'CNN Model is not loaded. Please train the model using train_model.py first.'
        }), 400

    classes = get_classes()
    saved_rel_path = None
    target_filepath = None

    # Check for sample image selection
    if 'sample_path' in request.form and request.form['sample_path']:
        sample_rel = request.form['sample_path'].strip().lstrip('/\\')
        target_filepath = os.path.normpath(os.path.join(BASE_DIR, 'static', sample_rel))
        if not os.path.exists(target_filepath):
            return jsonify({'success': False, 'error': f'Sample image {sample_rel} not found.'}), 404
        saved_rel_path = sample_rel.replace('\\', '/')

    # Check for uploaded file
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected for upload.'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file format. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Save uploaded file
        clean_name = secure_filename(file.filename)
        timestamp = int(time.time() * 1000)
        filename = f"upload_{timestamp}_{clean_name}"
        target_filepath = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(target_filepath)
        saved_rel_path = f"uploads/{filename}"

    else:
        return jsonify({'success': False, 'error': 'No image file or sample path provided.'}), 400

    # Perform Image Preprocessing and CNN Inference
    try:
        start_time = time.time()
        input_tensor, (orig_w, orig_h) = preprocess_image_file(target_filepath, img_size=IMG_SIZE)
        
        # Real model prediction
        predictions = model.predict(input_tensor, verbose=0)[0]
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        pred_idx = int(np.argmax(predictions))
        predicted_class = classes[pred_idx]
        confidence = float(predictions[pred_idx])
        confidence_percent = round(confidence * 100, 2)

        # Probabilities breakdown for all 4 classes
        probabilities = {}
        for idx, cls_name in enumerate(classes):
            prob_val = float(predictions[idx]) if idx < len(predictions) else 0.0
            probabilities[cls_name] = {
                'probability': round(prob_val, 4),
                'percentage': round(prob_val * 100, 2)
            }

        # Cell metadata
        info = CELL_INFO.get(predicted_class, {
            'name': predicted_class,
            'type': 'White Blood Cell',
            'morphology': 'Leukocyte structure',
            'normal_range': 'N/A',
            'clinical_function': 'Immune response',
            'badge_color': '#3b82f6'
        })

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence_percent,
            'confidence_raw': confidence,
            'probabilities': probabilities,
            'cell_info': info,
            'image_url': url_for('static', filename=saved_rel_path),
            'image_meta': {
                'original_dimensions': f"{orig_w} × {orig_h}",
                'preprocessed_dimensions': f"{IMG_SIZE[0]} × {IMG_SIZE[1]} × 3",
                'normalized_range': '[0.0, 1.0]',
                'inference_time_ms': inference_time_ms
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed during processing: {str(e)}'
        }), 500

@app.route('/api/metrics')
def get_metrics_api():
    """Return JSON evaluation metrics."""
    metrics = load_metrics()
    if metrics:
        return jsonify({'success': True, 'metrics': metrics})
    return jsonify({'success': False, 'error': 'Metrics not generated yet.'}), 404

if __name__ == '__main__':
    print("Starting Blood Cell Image Classification Flask Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)

