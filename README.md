# Deepfake Detection Project

This project aims to detect deepfake images of knee osteoarthritis scans by classifying them as either Real or Fake. It uses a deep learning model built on **EfficientNet-B3** and provides a user-friendly interface powered by **Gradio** for real-time predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Using the Gradio Interface](#using-the-gradio-interface)
- [Results](#results)
- [Dataset Organization](#dataset-organization)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contributing](#contributing)
- [Author](#author)

## Project Overview

Deepfake technology poses significant risks in various domains, including medical imaging. This project focuses on identifying whether knee osteoarthritis scans are authentic or generated by deepfake techniques.

The system includes:
- A robust classifier based on **EfficientNet-B3**.
- Evaluation tools to measure model performance using metrics like accuracy and confusion matrices.
- A drag-and-drop **Gradio interface** for real-time image classification.

## Features
- **Training Pipeline**: Includes data loading, augmentation, and model training scripts.
- **Evaluation Metrics**: Outputs classification accuracy, confusion matrix, and detailed classification reports.
- **Gradio Interface**: User-friendly interface for real-time predictions with drag-and-drop support.
- **Interactive Visualization**: Generates a confusion matrix and visualizes evaluation metrics.

## Installation

### Prerequisites
- **Python 3.8** or higher
- GPU-enabled environment (recommended but optional)
- Required Python libraries listed in `requirements.txt`

### Steps to Set Up the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/deepfake-detection-project.git
   cd deepfake-detection-project
   
2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

The project files and directories are organized as follows:

```plaintext
Deepfake_Detection_Project/
│
├── data/                         # Dataset folder
│   ├── real_images/              # Folder containing real images
│   └── fake_images/              # Folder containing deepfake images
│
├── src/                          # Source code for the project
│   ├── data_loader.py            # Data loading and preprocessing script
│   ├── train.py                  # Script for training the model
│   ├── model.py                  # Model definition (EfficientNet-B3)
│   ├── evaluate.py               # Script for evaluating the model
│   ├── inference.py              # Script for single image predictions
│   ├── config.py                 # Project configuration file
│
├── models/                       # Folder for saved models
│   ├── checkpoints/              # Checkpoints during training
│   └── final_model.pth           # Trained model for deployment
│
├── notebooks/                    # Jupyter notebooks for exploratory data analysis
│
├── results/                      # Results and evaluation logs
│   └── evaluation/               # Confusion matrix and evaluation metrics
│
├── app.py                        # Gradio interface for real-time predictions
├── README.md                     # Documentation for the project
└── requirements.txt              # Python dependencies

## Usage

### 1. Training the Model

Run the following script to train the model:

```bash
python src/train.py
```

- Input: Dataset organized into real_images and fake_images.
- Output: The trained model saved as models/final_model.pth.

### 2. Evaluating the Model

Evaluate the trained model on the validation set:

```bash
python src/evaluate.py
```

**Output:**
- Confusion matrix saved as results/evaluation/confusion_matrix.png.
- Detailed classification report printed to the console.

  
### 3. Using the Gradio Interface

Launch the Gradio app for real-time predictions:

```bash
python app.py
```

Open the app in your browser at: http://127.0.0.1:7860. Drag and drop an image or upload it to classify it as Real or Fake.

## Results

- **Confusion Matrix:** Saved in results/evaluation/confusion_matrix.png.
- **Classification Report:** Includes precision, recall, F1-score, and accuracy metrics for the validation set.

### example result ###
![confusion_matrix](https://github.com/user-attachments/assets/104767be-bfaf-4417-9d05-996eba55fe83)

## Dataset Organization

```plaintext
data/
├── real_images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake_images/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## Technologies Used
- **PyTorch:** Deep learning framework for training the classifier.
- **Torchvision:** Pre-trained models and transformations.
- **Gradio:** Interactive user interface for predictions.
- **Matplotlib & Seaborn:** Visualization of results.
- **Scikit-learn:** Evaluation metrics and confusion matrix generation.

## Author
Prashant singh
Contact: [prashantsingha96@gmail.com]
