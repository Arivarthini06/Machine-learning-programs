# Blood Cell Image Classification Using Machine Learning

A complete, working machine learning and deep learning project designed for medical image classification of white blood cells (leukocytes) into four primary classes using a Convolutional Neural Network (CNN) and a local Flask web application.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Target Classes](#target-classes)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Technology Stack](#technology-stack)
- [Project Directory Structure](#project-directory-structure)
- [Dataset Organization](#dataset-organization)
- [Image Preprocessing](#image-preprocessing)
- [CNN Model Architecture](#cnn-model-architecture)
- [Model Training & Evaluation](#model-training--evaluation)
- [Installation & Setup](#installation--setup)
- [How to Run the Application](#how-to-run-the-application)
- [Project Record Screenshots Guide](#project-record-screenshots-guide)

---

## 🎯 Project Overview

In clinical hematology, examining peripheral blood smears to classify white blood cells is essential for diagnosing infections, immune deficiencies, allergies, and hematological disorders. This application automates blood cell classification by taking a microscopic blood cell image and predicting its specific leukocyte type along with a confidence percentage.

### Key Features
- **Authentic Dataset**: 1,800 microscopic blood cell images across 4 balanced classes.
- **Dynamic Class Detection**: Automatically scans and detects classes from the `dataset/` folder.
- **Real-Time CNN Inference**: Preprocesses and classifies images in milliseconds.
- **Interactive Web Interface**: Clean medical laboratory UI with live drag-and-drop upload and 1-click sample testing.
- **Complete Evaluation Suite**: Real accuracy/loss curves, confusion matrix heatmap, and classification report.
- **100% Offline & Local**: Runs entirely on the local computer with no cloud dependencies or API keys.

---

## 🔬 Target Classes

The system classifies blood cell images into one of four leukocyte types:

| Class | Type | Distinguishing Microscopic Morphology | Primary Biological Function |
|---|---|---|---|
| **Eosinophil** | Granulocyte | Prominent bilobed nucleus with coarse reddish-orange granules. | Combats parasitic infections; mediates allergic reactions. |
| **Lymphocyte** | Agranulocyte | Large, dense spherical nucleus with minimal rim of pale blue cytoplasm. | Effector of adaptive immunity (T-cells, B-cells, NK cells). |
| **Monocyte** | Agranulocyte | Largest leukocyte with a kidney- or horseshoe-shaped nucleus. | Precursor to macrophages and dendritic cells for phagocytosis. |
| **Neutrophil** | Granulocyte | Multilobed nucleus (3–5 lobes) with fine neutral/lilac granules. | Primary first responder against acute bacterial infections. |

---

## 🔄 System Architecture & Workflow

```text
┌───────────────────────────────┐
│   Blood Cell Image Dataset    │ (1,800 images in dataset/)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│      Image Preprocessing      │ (RGB conversion, 128×128 resize, [0, 1] normalization)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│    Feature Learning (CNN)     │ (Conv2D, ReLU activation, MaxPooling2D)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│        Train CNN Model        │ (Adam optimizer, Sparse Categorical Crossentropy)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│        Evaluate Model         │ (Accuracy & Loss plots, Confusion Matrix, Metrics)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│   Upload Blood Cell Image     │ (Web UI drag-and-drop or sample selection)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│   Predict Cell Type Result    │ (Predicted class + real-time confidence %)
└───────────────────────────────┘
```

---

## 💻 Technology Stack

- **Backend**: Python 3.x, Flask
- **Deep Learning / ML**: TensorFlow, Keras, Scikit-learn, NumPy, Pandas
- **Image Processing**: Pillow (PIL), OpenCV-compatible arrays
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (Medical Lab Theme), JavaScript (Fetch API)

---

## 📁 Project Directory Structure

```text
blood-cell-image-classification/
│
├── app.py                      # Flask web application & prediction endpoints
├── train_model.py              # CNN model training, evaluation & plot generator
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
│
├── dataset/                    # Dataset directory with 4 class subfolders
│   ├── Eosinophil/             # 450 Eosinophil images (.jpg)
│   ├── Lymphocyte/             # 450 Lymphocyte images (.jpg)
│   ├── Monocyte/               # 450 Monocyte images (.jpg)
│   └── Neutrophil/             # 450 Neutrophil images (.jpg)
│
├── models/
│   └── blood_cell_cnn.keras    # Saved trained Keras CNN model
│
├── static/
│   ├── style.css               # Clean laboratory UI stylesheet
│   ├── uploads/                # Temporary user uploaded images
│   ├── plots/                  # Generated evaluation plots & metrics
│   │   ├── accuracy_plot.png
│   │   ├── loss_plot.png
│   │   ├── confusion_matrix.png
│   │   ├── dataset_distribution.png
│   │   ├── preprocessing_sample.png
│   │   └── metrics.json
│   └── sample_images/          # Demo blood cell samples for 1-click testing
│       ├── Eosinophil/
│       ├── Lymphocyte/
│       ├── Monocyte/
│       └── Neutrophil/
│
└── templates/
    └── index.html              # Main web interface template
```

---

## ⚙️ Image Preprocessing

Before passing images into the CNN:
1. **RGB Formatting**: Converts images to 3-channel standard RGB, removing palette channels.
2. **Spatial Resizing**: Resizes every input image to a fixed **$128 \times 128$** resolution using bilinear interpolation.
3. **Pixel Normalization**: Scales integer pixel values $[0, 255]$ into floating-point numbers in the range **$[0.0, 1.0]$** ($X / 255.0$).
4. **Label Encoding**: Encodes classes numerically:
   - `0`: Eosinophil
   - `1`: Lymphocyte
   - `2`: Monocyte
   - `3`: Neutrophil
5. **Stratified Splitting**: 70% Training, 15% Validation, 15% Test.

---

## 🧠 CNN Model Architecture

The primary model is a Convolutional Neural Network (CNN) built with Keras:

| Layer | Type | Specifications | Activation |
|---|---|---|---|
| 1 | Input | $(128, 128, 3)$ tensor | — |
| 2 | Conv2D | 32 filters, $(3 \times 3)$, padding='same' | ReLU |
| 3 | MaxPooling2D | Pool size $(2 \times 2)$ | — |
| 4 | Conv2D | 64 filters, $(3 \times 3)$, padding='same' | ReLU |
| 5 | MaxPooling2D | Pool size $(2 \times 2)$ | — |
| 6 | Conv2D | 128 filters, $(3 \times 3)$, padding='same' | ReLU |
| 7 | MaxPooling2D | Pool size $(2 \times 2)$ | — |
| 8 | Conv2D | 128 filters, $(3 \times 3)$, padding='same' | ReLU |
| 9 | MaxPooling2D | Pool size $(2 \times 2)$ | — |
| 10 | Flatten | Flattens 2D feature maps to 1D vector | — |
| 11 | Dense | 128 units | ReLU |
| 12 | Dropout | Rate = 0.3 (Regularization) | — |
| 13 | Dense (Output) | 4 units | Softmax |

- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam(learning_rate=0.0003)`
- **Evaluation Metric**: `accuracy`

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Required Dependencies
Open PowerShell or Command Prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run the Application

### Step 1: Train the CNN Model (Optional if already trained)
The project comes with a pre-trained model and generated plots. To retrain the CNN from scratch on the dataset:

```bash
python train_model.py --epochs 10
```

This will:
- Load all images from `dataset/`.
- Perform image preprocessing.
- Train the CNN for 10 epochs.
- Evaluate on unseen test images.
- Generate and save `accuracy_plot.png`, `loss_plot.png`, `confusion_matrix.png`, `dataset_distribution.png`, and `preprocessing_sample.png` into `static/plots/`.
- Save the trained model to `models/blood_cell_cnn.keras`.

### Step 2: Start the Flask Web Application
Run the Flask server:

```bash
python app.py
```

### Step 3: Open in Browser
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📸 Project Record Screenshots Guide

The web application is structured specifically to capture clear figures for lab manuals and project records:

- **Fig. 1 — Dataset Overview**: Shows the 4 leukocyte classes, sample counts (450 per class), and the class distribution bar chart.
- **Fig. 2 — Image Preprocessing**: Shows the Original microscopic image, Resized ($128 \times 128$) image, and Normalized $[0.0, 1.0]$ tensor transformation.
- **Fig. 3 — CNN Training**: Shows the CNN layer architecture table and the **Accuracy vs. Epochs** and **Loss vs. Epochs** curves.
- **Fig. 4 — Model Evaluation**: Shows Test Accuracy, Precision, Recall, F1-Score table, and the **Confusion Matrix Heatmap**.
- **Fig. 5 — Image Upload**: Shows the image upload drag-and-drop zone and image preview.
- **Fig. 6 — Prediction Result**: Shows the predicted cell type badge (e.g., *Neutrophil*), confidence percentage (e.g., *94.25%*), 4-class probability breakdown, and clinical details.
