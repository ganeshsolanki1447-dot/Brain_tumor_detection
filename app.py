from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs('uploads', exist_ok=True)

# Load trained model
try:
    model = load_model('brain_tumor_model.h5')
    print("Model loaded successfully!")
except:
    print("Model not found. Please run model_trainer.py first.")
    model = None

# Class names (adjust based on your dataset)
CLASS_NAMES = {
    0: 'glioma',
    1: 'meningioma', 
    2: 'no_tumor',
    3: 'pituitary'
}

TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'Malignant brain tumor arising from glial cells',
        'severity': 'High',
        'action': 'Seek immediate medical attention'
    },
    'meningioma': {
        'name': 'Meningioma', 
        'description': 'Usually benign tumor of the meninges',
        'severity': 'Low-Moderate',
        'action': 'Consult with neurosurgeon'
    },
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'description': 'Benign tumor of the pituitary gland',
        'severity': 'Low-Moderate', 
        'action': 'Endocrinology consultation recommended'
    },
    'no_tumor': {
        'name': 'No Tumor Detected',
        'description': 'Normal brain tissue',
        'severity': 'None',
        'action': 'Continue routine monitoring'
    }
}

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess and predict
        img = preprocess_image(filepath)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Get tumor information
        tumor_type = CLASS_NAMES[predicted_class]
        tumor_info = TUMOR_INFO.get(tumor_type, {})

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'prediction': tumor_type,
            'confidence': confidence * 100,
            'tumor_info': tumor_info,
            'all_probabilities': {
                CLASS_NAMES[i]: float(prob) * 100 
                for i, prob in enumerate(predictions[0])
            }
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
