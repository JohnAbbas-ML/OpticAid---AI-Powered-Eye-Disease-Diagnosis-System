import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
# Hybrid CNN Model with Attention Mechanism
class ChemicalImageClassifier(nn.Module):
    def __init__(self, num_classes, input_size=224):
        super().__init__()
        
        # CNN Layers with Progressive Depth
        self.cnn_layers = nn.Sequential(
            # Initial layers with increasing depth
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Self-Attention Mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # CNN Feature Extraction
        features = self.cnn_layers(x)
        
        # Self-Attention Mechanism
        attention_weights = self.attention(features)
        features_attended = features * attention_weights
        
        # Global Average Pooling
        pooled_features = self.gap(features_attended)
        
        # Classification
        return self.classifier(pooled_features)

app = Flask(__name__)

# Load the model
MODEL_PATH = 'eye_image_classifier.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = [
    'Cataract', 
    'Conjunctivitis', 
    'Eyelid', 
    'Normal',
    'Uveitis'
]

def load_model():
    """Load the pre-trained model"""
    model = ChemicalImageClassifier(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def transform_image(image_bytes):
    """Transform input image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    """Get model prediction"""
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        return predicted.item(), probabilities.cpu().numpy()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read file and predict
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    class_id, probabilities = get_prediction(tensor)
    
    # Prepare response
    response = {
        'predicted_class': CLASS_NAMES[class_id],
        'probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probabilities)}
    }
    
    return jsonify(response)

@app.route('/visualize', methods=['POST'])
def visualize_probabilities():
    """Create visualization of model probabilities"""
    probabilities = request.json.get('probabilities', {})
    
    plt.figure(figsize=(10, 6))
    plt.bar(probabilities.keys(), probabilities.values())
    plt.title('Disease Probability Distribution')
    plt.xlabel('Disease')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({'image': image_base64})

if __name__ == '__main__':
    # Load model when the app starts
    model = load_model()
    app.run(debug=True)