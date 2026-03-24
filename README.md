# AEROVISION 🛩️

**Data-driven Fighter Jet Classification System**

AEROVISION is a deep learning-based web application that identifies and classifies fighter jets from images using state-of-the-art computer vision technology. Built with EfficientNetV2S architecture, this system provides real-time classification of military aircraft with detailed technical specifications.

---

## 📋 Project Overview

AEROVISION leverages transfer learning and convolutional neural networks to classify fighter jets into 5 distinct classes:

- **B-2 Spirit** - USA Stealth Strategic Bomber
- **F-22 Raptor** - USA 5th Generation Air Superiority Fighter
- **Chengdu J-20** - China 5th Generation Stealth Fighter
- **Dassault Rafale** - France 4.5 Generation Multirole Fighter
- **Sukhoi Su-57** - Russia 5th Generation Stealth Multirole Fighter

The system provides instant classification results along with confidence scores and comprehensive technical information about each aircraft.

---

## 🚀 Technology Stack

### Backend & Machine Learning
- **Flask** - Web framework for serving the application
- **TensorFlow 2.16.1** - Deep learning framework
- **Keras 3.0+** - High-level neural networks API
- **EfficientNetV2S** - Pre-trained CNN model (ImageNet-21k weights)
  - 83.9% top-1 accuracy baseline
  - Native 384×384 resolution for fine detail capture
  - Built-in preprocessing pipeline

### Data Processing & Analysis
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and CSV handling
- **Pillow** - Image loading and preprocessing
- **OpenCV** - Computer vision operations
- **scikit-learn** - Model evaluation metrics and class balancing

### Frontend
- **HTML5/CSS3** - Responsive UI design
- **JavaScript** - Interactive client-side functionality
- **Custom CSS** - Styled pages (home.css, capabilities.css, docs.css, about.css)

### Model Architecture
- **Base Model**: EfficientNetV2S (frozen backbone)
- **Custom Classifier Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(512) + Dropout(0.5) + L2 Regularization
  - Dense(256) + Dropout(0.4) + L2 Regularization
  - Dense(5, softmax) - Output layer
- **Training Strategy**: Two-phase training
  - Phase 1: Train classifier head (20 epochs, LR=1e-3)
  - Phase 2: Fine-tune top 30 backbone layers (30 epochs, LR=1e-5)

---

## 🗺️ Site Map

The application features a comprehensive multi-page interface:

### 1. **Home Page** (`/`)
   - Landing page with project introduction
   - Hero section with call-to-action
   - Navigation to all features

### 2. **Classification Page** (`/classify`)
   - Image upload interface
   - Real-time jet classification
   - Confidence scores display
   - All class probability scores
   - Detailed aircraft information retrieval

### 3. **Capabilities Page** (`/capabilities`)
   - System features and capabilities
   - Technology overview
   - Model performance metrics
   - Use cases and applications

### 4. **About Page** (`/about-us`)
   - Project background and motivation
   - Team information
   - Development journey
   - Contact information

### 5. **Documentation Page** (`/documents`)
   - Technical documentation
   - API reference
   - Model architecture details
   - Downloadable resources

### Components
- **Navbar** - Consistent navigation across all pages
- **Footer** - Site-wide footer with links and information
- **HUD** - Heads-up display elements for enhanced UI

---

## 📊 Dataset & Training Details

### Dataset Structure
```
data/Fighter_jet_final_data/
├── Train/       (350 images - 70 per class)
│   ├── B2/
│   ├── F22/
│   ├── J20/
│   ├── Rafale/
│   └── Su57/
├── Validation/  (50 images - 10 per class)
└── Test/        (100 images - 20 per class)
```

### Data Augmentation
- Rotation: ±20°
- Width/Height shift: 10%
- Shear transformation: 10%
- Zoom range: 15%
- Horizontal flip
- Brightness variation: 85-115%

### Training Configuration
- **Image Size**: 384×384 pixels
- **Batch Size**: 8
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- **Class Balancing**: Computed class weights for balanced training

### Model Performance
- **Test Accuracy**: 75.00%
- **Validation Accuracy**: 78.00% (best)
- **Per-Class Performance**:
  - B2 Spirit: 94% precision, 75% recall
  - F-22 Raptor: 53% precision, 85% recall
  - J-20: 100% precision, 70% recall
  - Rafale: 86% precision, 90% recall
  - Su-57: 65% precision, 55% recall

---

## ⚠️ Known Limitations & Weak Points

### 1. **Limited Training Data**
   - Model trained on only **350 training images** (70 per class)
   - Small dataset limits the model's ability to generalize
   - **Maximum accuracy achieved: ~79%** on validation set

### 2. **Overfitting to Training Distribution**
   - Model may struggle with aircraft images that differ significantly from training data
   - Variations in angle, lighting, background, or distance can affect accuracy
   - Limited exposure to diverse real-world scenarios

### 3. **False Classification of Random Images**
   - **Critical Issue**: The model will classify ANY input image into one of the 5 jet classes
   - No "unknown" or "not a jet" category exists
   - Random objects, animals, or unrelated images will be forced into one of the 5 classes
   - Confidence scores may still appear high even for incorrect classifications

### 4. **Class Imbalance in Performance**
   - Su-57 has notably lower performance (55% recall)
   - F-22 has lower precision (53%) leading to false positives
   - Performance varies significantly across different aircraft types

### 5. **Resolution and Detail Requirements**
   - Model expects 384×384 pixel images
   - Very low-resolution or heavily compressed images may degrade performance
   - Extreme angles or partial views may not be recognized correctly

### Recommendations for Users
- Use clear, well-lit images of fighter jets
- Ensure the aircraft is the primary subject in the image
- Avoid uploading non-aircraft images (results will be unreliable)
- Consider confidence scores - lower scores may indicate uncertain predictions
- Best results with side or three-quarter view angles

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 - 3.12
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or Download the Project**
   ```bash
   cd path/to/The_Jet_Project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv virtual_env
   ```

3. **Activate Virtual Environment**
   
   **Windows:**
   ```bash
   virtual_env\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source virtual_env/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Model Files**
   Ensure these model files exist in the project root:
   - `fighter_jet_classifier.h5` (main model)
   - `best_model_phase1.h5` (optional - phase 1 checkpoint)
   - `best_model_phase2.h5` (optional - phase 2 checkpoint)

6. **Verify Data Files**
   Ensure the CSV file exists:
   - `data/fighter_jets_info.csv`

7. **Run the Application**
   ```bash
   python app.py
   ```

8. **Access the Application**
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

### Alternative: Run with Debug Mode
```bash
python app.py
```
The Flask app runs in debug mode by default (see `app.py`).

---

## 🔧 API Endpoints

### `GET /`
- **Description**: Home page
- **Returns**: Landing page HTML

### `GET /classify`
- **Description**: Classification interface
- **Returns**: Upload and classification page HTML

### `POST /predict`
- **Description**: Classify uploaded jet image
- **Request**: Multipart form data with image file
- **Response**: JSON
  ```json
  {
    "prediction": "F22",
    "confidence": 95.87,
    "all_scores": {
      "B2": 0.05,
      "F22": 95.87,
      "J20": 2.15,
      "Rafale": 1.23,
      "Su57": 0.70
    }
  }
  ```

### `GET /jet_info/<class_name>`
- **Description**: Get detailed information about a specific jet
- **Parameters**: class_name (B2, F22, J20, Rafale, Su57)
- **Response**: JSON with aircraft specifications

### `GET /capabilities`
- **Description**: System capabilities page
- **Returns**: Capabilities page HTML

### `GET /about-us`
- **Description**: About page
- **Returns**: About page HTML

### `GET /documents`
- **Description**: Documentation page
- **Returns**: Documentation page HTML

### `GET /download/<filename>`
- **Description**: Download documentation files
- **Returns**: File download

---

## 📈 Future Improvements

- Expand dataset with more diverse images (different angles, weather conditions, backgrounds)
- Implement "unknown" class detection for non-jet images
- Add more fighter jet classes (F-35, Eurofighter Typhoon, etc.)
- Improve model architecture for better generalization
- Implement real-time video classification
- Add data augmentation techniques for better robustness
- Deploy to cloud platforms (AWS, Azure, GCP)
- Create mobile application version
- Add multi-language support

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Contributors

Developed as part of an AI/ML learning project focused on computer vision and deep learning applications.

---

## 📸 Model Output Examples

The `Output_Images/` folder contains sample prediction outputs from the trained model, demonstrating the classification results for different fighter jets:

- **B2_OP.png** - B-2 Spirit classification output
- **J20_OP.png** - Chengdu J-20 classification output
- **Rafale_OP.png** - Dassault Rafale classification output
- **SU57_OP.png** - Sukhoi Su-57 classification output

These images showcase the model's prediction confidence scores and visual results for each aircraft class. You can view these examples to understand how the system presents classification results to users.

---

**AEROVISION** - Bringing AI-powered aircraft recognition to your fingertips! ✈️🚀
