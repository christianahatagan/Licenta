"""
Flask Backend for Mammography Detection App
Handles image upload, preprocessing, and YOLOv8 detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from ultralytics import YOLO
import json

# Import our preprocessing functions
import sys
sys.path.append(str(Path(__file__).parent))
from preprocessing import preprocess_image
from config import Config

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SECRET_KEY'] = 'mammography-detection-2026-licenta-christiana'

# Ensure upload folder exists
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def numpy_to_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 string for JSON transfer"""
    if len(img.shape) == 2:  # Grayscale
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img
    
    _, buffer = cv2.imencode('.png', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def load_model(method: str, view: str):
    """
    Load YOLOv8 model for specific preprocessing method and view.
    
    Args:
        method: preprocessing method name (e.g., 'test', 'gaussian_clahe')
        view: 'cc', 'mlo', or 'cc_mlo'
    
    Returns:
        YOLO model object or None if not found
    """
    model_path = Config.MODELS_FOLDER / f"{view}_{method}.pt"
    
    if not model_path.exists():
        print(f"[WARNING] Model not found: {model_path}")
        return None
    
    try:
        model = YOLO(str(model_path))
        print(f"[INFO] Loaded model: {model_path.name}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return None


def run_detection(model, img: np.ndarray, conf_threshold: float = 0.25):
    """
    Run YOLOv8 detection on preprocessed image.
    
    Returns:
        List of detections with bounding boxes and confidence scores
    """
    if model is None:
        return []
    
    try:
        results = model.predict(
            source=img,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0]),
                    'class_name': 'Malignant' if int(box.cls[0]) == 1 else 'Benign'
                }
                detections.append(detection)
        
        return detections
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return []


def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on image"""
    img_draw = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        class_name = det['class_name']
        
        # Color: Red for Malignant, Green for Benign
        color = (0, 0, 255) if class_name == 'Malignant' else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_draw


# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         methods=Config.PREPROCESSING_METHODS,
                         views=Config.VIEWS,
                         performance=Config.MODEL_PERFORMANCE)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and preprocessing
    Returns preprocessing steps for visualization
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, DCM'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Config.UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Read image
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Get selected preprocessing method
        method = request.form.get('method', 'test')
        
        # Preprocess with steps
        result = preprocess_image(img, method=method, return_steps=True)
        
        # Convert steps to base64 for frontend
        steps_b64 = {}
        for step_name, step_img in result['steps'].items():
            steps_b64[step_name] = numpy_to_base64(step_img)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'steps': steps_b64,
            'method': method
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect', methods=['POST'])
def detect():
    """
    Run YOLOv8 detection on preprocessed image
    """
    try:
        data = request.json
        filename = data.get('filename')
        method = data.get('method')
        view = data.get('view', 'cc_mlo')
        conf_threshold = float(data.get('confidence', 0.25))
        
        # Read and preprocess image
        filepath = Config.UPLOAD_FOLDER / filename
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        
        result = preprocess_image(img, method=method, return_steps=False)
        preprocessed_img = result['final']
        
        # Load model
        model = load_model(method, view)
        if model is None:
            return jsonify({
                'error': f'Model not found: {view}_{method}.pt',
                'available_models': 'Please download models from Google Drive (see README)'
            }), 404
        
        # Run detection
        detections = run_detection(model, preprocessed_img, conf_threshold)
        
        # Draw detections
        img_with_boxes = draw_detections(preprocessed_img, detections)
        
        # Get performance metrics for this combination
        metrics = Config.MODEL_PERFORMANCE.get(view, {}).get(method, {})
        
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'image_with_boxes': numpy_to_base64(img_with_boxes),
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get all model performance metrics for dashboard
    """
    return jsonify({
        'performance': Config.MODEL_PERFORMANCE,
        'methods': Config.PREPROCESSING_METHODS,
        'views': Config.VIEWS
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


# ═══════════════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" MAMMOGRAPHY DETECTION APP - Starting...")
    print("="*60)
    print(f" Upload folder: {Config.UPLOAD_FOLDER}")
    print(f" Models folder: {Config.MODELS_FOLDER}")
    print(f" Available methods: {list(Config.PREPROCESSING_METHODS.keys())}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)