# Brain Tumor Detection Application

This is a fully functional machine learning application that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN).

## Features

- Real CNN model trained on medical images
- Image upload and preprocessing
- 4-class classification (Glioma, Meningioma, Pituitary, No Tumor)
- Confidence scores and medical recommendations
- Professional medical UI

## Setup Instructions

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Kaggle API (for dataset):**
```bash
# Go to Kaggle -> Account -> API -> Create New API Token
# Place kaggle.json in ~/.kaggle/ directory
```

3. **Train the model:**
```bash
python model_trainer.py
```

4. **Run the application:**
```bash
python app.py
```

5. **Access the application:**
Open your browser and go to: http://localhost:5001

## Usage

1. Upload an MRI brain scan image (JPEG, PNG)
2. Click "Analyze Image"
3. View the results with confidence scores
4. Get medical recommendations based on the prediction

## Model Information

- Architecture: CNN with multiple convolutional layers
- Input size: 224x224x3
- Classes: Glioma, Meningioma, Pituitary, No Tumor
- Training dataset: Brain tumor MRI dataset from Kaggle

## Medical Disclaimer

This tool is for educational and research purposes only. Always consult qualified medical professionals for diagnosis and treatment decisions.
