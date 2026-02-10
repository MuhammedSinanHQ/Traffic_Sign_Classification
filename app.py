import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load model at startup
MODEL_PATH = 'model.h5'
model = None

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Class names for traffic signs (43 classes)
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield',
    'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             error="Model not loaded. Please check server logs.")
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', 
                             error="No file uploaded. Please select an image.")
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return render_template('index.html', 
                             error="No file selected. Please choose an image.")
    
    try:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to (50, 50)
        image = image.resize((50, 50))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array / 255.0
        
        # Expand dimensions to match model input shape
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        # Get class name
        predicted_label = class_names[predicted_class]
        
        return render_template('index.html', 
                             prediction=predicted_label,
                             confidence=f"{confidence:.2f}",
                             class_id=predicted_class)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error processing image: {str(e)}")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
