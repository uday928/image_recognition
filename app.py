import os
import csv
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = 'fighter_jet_classifier.h5'
model = load_model(MODEL_PATH)

CLASS_NAMES = ['B2', 'F22', 'J20', 'Rafale', 'Su57']
IMG_SIZE    = (384, 384)   # must match training — (224,224) if you used B0, (384,384) for V2S

CSV_PATH = 'data/fighter_jets_info.csv'

CLASS_TO_CSV = {
    'B2':     'B-2 Spirit',
    'F22':    'F-22 Raptor',
    'J20':    'Chengdu J-20',
    'Rafale': 'Dassault Rafale',
    'Su57':   'Sukhoi Su-57'
}

def load_jet_info():
    jets = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jets[row['jet_name']] = row
    return jets

JET_INFO = load_jet_info()

# ═══════════════════════════════════════════════════════════════
# ROUTE CHANGE: Root route now serves landing page instead of classifier
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')  # Changed from 'index.html'

# ═══════════════════════════════════════════════════════════════
# NEW ROUTE: Added /classify route for the main classifier page
# ═══════════════════════════════════════════════════════════════
@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/jet_info/<class_name>')
def jet_info(class_name):
    csv_name = CLASS_TO_CSV.get(class_name)
    if not csv_name or csv_name not in JET_INFO:
        return jsonify({'error': 'Jet info not found'}), 404
    return jsonify(JET_INFO[csv_name])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    temp_path = os.path.join('static', 'temp_upload.jpg')
    file.save(temp_path)

    img       = image.load_img(temp_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)          # raw 0-255, NO /255.0
    img_array = np.expand_dims(img_array, axis=0) # shape: (1, 384, 384, 3)

    predictions     = model.predict(img_array)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(predictions[predicted_index]) * 100

    all_scores = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 2),
        'all_scores': all_scores
    })

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)