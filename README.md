# OpticAid - AI Powered Eye Disease Detection System 👁️🤖

Welcome to **OpticAid**, a deep learning-based Flask web application for detecting eye diseases from images using a hybrid CNN with self-attention mechanism.

---

## 🛠 Project Structure

```
├── app.py                  # Flask backend to serve the model and API
├── eye_image_classifier.pth # Trained model weights
├── eyedisease-model.ipynb   # Model training and evaluation notebook
├── Data/                    # Image dataset
├── templates/
│   └── index.html           # Frontend (HTML + TailwindCSS + Chart.js)
└── README.md                # Project description
```

---

## 🚀 How to Run the Project Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/optic-aid-eye-disease-classifier.git
   cd optic-aid-eye-disease-classifier
   ```

2. **Create a Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Make Sure You Have the Model File**
   - Place `eye_image_classifier.pth` in the project root directory.

5. **Run the Flask App**
   ```bash
   python app.py
   ```

6. **Visit the Web App**
   - Open your browser and go to: `http://127.0.0.1:5000/`

---

## 📸 How it Works

- Upload an eye image.
- The trained CNN model predicts the possible eye disease:
  - Cataract
  - Conjunctivitis
  - Eyelid Disease
  - Normal (Healthy)
  - Uveitis
- Displays the prediction result and probability distribution as a chart.

---

## 🧠 Model Architecture

- **Custom CNN Layers** with progressive depth
- **Self-Attention Mechanism** after feature extraction
- **Global Average Pooling** to reduce feature dimensions
- **Fully Connected Layers** for classification
- **Training Strategy:**
  - Data augmentation
  - Early stopping
  - Learning rate scheduler
  - Evaluation using classification reports and confusion matrix

---

## 🖼 Frontend Features

- Built with **TailwindCSS** for clean UI.
- **Chart.js** for visualizing prediction probabilities.
- Mobile responsive design.

---

## 📦 Requirements

- Python 3.7+
- Flask
- Torch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- Pillow

> Install all at once:
> ```bash
> pip install flask torch torchvision numpy matplotlib seaborn scikit-learn Pillow
> ```

---

## 📊 Results

- Achieved high validation accuracy during training.
- Effective across 5 classes of eye diseases.
- Visualized confusion matrix and classification report during evaluation.

---


## 🌟 Acknowledgments

- Special thanks to the open datasets and PyTorch community.
- Inspired by real-world needs for accessible eye disease detection.

---
