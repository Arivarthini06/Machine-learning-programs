"""
Blood Cell Image Classification Using Machine Learning
======================================================
Convolutional Neural Network (CNN) Training and Evaluation Pipeline.

Detects 4 blood cell classes from dataset/:
- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Global Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
MODEL_PATH = os.path.join(MODELS_DIR, "blood_cell_cnn.keras")
METRICS_PATH = os.path.join(PLOTS_DIR, "metrics.json")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
RANDOM_STATE = 42

def setup_dirs():
    """Create all required directory structures."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "uploads"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "sample_images"), exist_ok=True)

def detect_classes(dataset_path=DATASET_DIR):
    """Detect classes dynamically from the folder structure."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    classes = [
        d for d in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')
    ]
    if not classes:
        raise ValueError(f"No subdirectories found in '{dataset_path}'.")
    return classes

def load_and_preprocess(dataset_path=DATASET_DIR, img_size=IMG_SIZE):
    """
    1. Reads raw images from each class directory.
    2. Resizes images to (128, 128).
    3. Converts to RGB format.
    4. Normalizes pixel values [0.0, 1.0].
    5. Returns image tensor, numerical labels, class counts, and sample demo images.
    """
    classes = detect_classes(dataset_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    images = []
    labels = []
    class_counts = {}
    sample_images = {}

    print("=" * 65)
    print("STEP 1: LOADING & PREPROCESSING BLOOD CELL DATASET")
    print("=" * 65)
    print(f"Target directory : {dataset_path}")
    print(f"Detected classes : {classes}")

    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[c] = len(files)
        print(f"  • {c:<12}: {len(files)} images")

        for idx, fname in enumerate(files):
            fpath = os.path.join(c_dir, fname)
            try:
                with Image.open(fpath) as img:
                    rgb_img = img.convert('RGB')
                    
                    if c not in sample_images:
                        sample_images[c] = {
                            'original': np.array(rgb_img),
                            'path': fpath,
                            'name': fname
                        }
                    
                    resized = rgb_img.resize(img_size, Image.Resampling.BILINEAR)
                    norm_arr = np.asarray(resized, dtype=np.float32) / 255.0
                    images.append(norm_arr)
                    labels.append(class_to_idx[c])
            except Exception as err:
                print(f"    [Warning] Failed loading {fpath}: {err}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"\nTotal Loaded Images : {X.shape[0]}")
    print(f"Input Tensor Shape  : {X.shape} (N, H, W, C)")
    print(f"Labels Shape        : {y.shape}")
    print("=" * 65)
    return X, y, classes, class_counts, sample_images

def build_cnn(input_shape=(128, 128, 3), num_classes=4):
    """
    Constructs a Convolutional Neural Network (CNN) as per requirements:
    - Conv2D layers with ReLU
    - MaxPooling2D
    - Flatten
    - Dense layer with Dropout
    - Softmax output layer
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_1'),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2d_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_2'),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2d_3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_3'),
        
        # Conv Block 4
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2d_4'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_4'),
        
        # Classifier Head
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense_feature'),
        layers.Dropout(0.3, name='dropout_regularization'),
        layers.Dense(num_classes, activation='softmax', name='prediction_softmax')
    ], name='BloodCellCNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_preprocessing(sample_images, classes):
    """Generates Figure 2: Image Preprocessing pipeline visualization."""
    first_cls = classes[0]
    sample = sample_images.get(first_cls)
    if not sample:
        return
    
    orig = sample['original']
    orig_h, orig_w, _ = orig.shape
    resized_pil = Image.fromarray(orig).resize(IMG_SIZE, Image.Resampling.BILINEAR)
    resized_np = np.array(resized_pil)
    normalized_np = resized_np / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150)
    
    # 1. Original
    axes[0].imshow(orig)
    axes[0].set_title(f"1. Original Microscopic Image\nClass: {first_cls} ({orig_w}x{orig_h} RGB)", 
                      fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('on')
    axes[0].tick_params(labelsize=8)

    # 2. Resized
    axes[1].imshow(resized_np)
    axes[1].set_title(f"2. Resized Image\nFixed Dimension: {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB", 
                      fontsize=11, fontweight='bold', pad=10)
    axes[1].axis('on')
    axes[1].tick_params(labelsize=8)

    # 3. Normalized Representation
    axes[2].imshow(normalized_np)
    axes[2].set_title("3. Normalized Tensor\nPixel Range: [0.0, 1.0]", 
                      fontsize=11, fontweight='bold', pad=10)
    axes[2].axis('on')
    axes[2].tick_params(labelsize=8)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "preprocessing_sample.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Preprocessing pipeline plot -> {out_file}")

def plot_dataset_dist(class_counts):
    """Generates Figure 1: Dataset class distribution plot."""
    plt.figure(figsize=(8, 4.5), dpi=150)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#1d4ed8', '#0d9488', '#d97706', '#dc2626']
    
    bars = plt.bar(classes, counts, color=colors[:len(classes)], edgecolor='#0f172a', linewidth=1.2, width=0.55)
    plt.title("Blood Cell Dataset - Samples per Class", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Cell Type (WBC)", fontsize=11, fontweight='bold')
    plt.ylabel("Image Count", fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.015), f"{int(yval)}", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "dataset_distribution.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Dataset distribution plot  -> {out_file}")

def plot_training(history):
    """Generates Figure 3: Accuracy vs Epochs & Loss vs Epochs plots."""
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # 1. Accuracy
    plt.figure(figsize=(7.5, 4.5), dpi=150)
    plt.plot(epochs, history.history['accuracy'], 'o-', color='#2563eb', linewidth=2.2, label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 's-', color='#16a34a', linewidth=2.2, label='Validation Accuracy')
    plt.title("CNN Model Accuracy vs Epochs", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Epoch", fontsize=11, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    acc_path = os.path.join(PLOTS_DIR, "accuracy_plot.png")
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()

    # 2. Loss
    plt.figure(figsize=(7.5, 4.5), dpi=150)
    plt.plot(epochs, history.history['loss'], 'o-', color='#dc2626', linewidth=2.2, label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 's-', color='#f59e0b', linewidth=2.2, label='Validation Loss')
    plt.title("CNN Model Loss vs Epochs", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Epoch", fontsize=11, fontweight='bold')
    plt.ylabel("Loss", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(PLOTS_DIR, "loss_plot.png")
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()

    print(f"  [Saved] Accuracy vs Epochs plot    -> {acc_path}")
    print(f"  [Saved] Loss vs Epochs plot        -> {loss_path}")

def plot_confusion_matrix(cm, classes):
    """Generates Figure 4: Confusion Matrix Heatmap."""
    plt.figure(figsize=(7.5, 6), dpi=150)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        linewidths=1.2,
        linecolor='#e2e8f0',
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    plt.title("Confusion Matrix (Test Set Evaluation)", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Actual True Label", fontsize=11, fontweight='bold', labelpad=10)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Confusion Matrix plot      -> {cm_path}")

def train_and_evaluate(epochs=DEFAULT_EPOCHS, force=False):
    """Full execution pipeline."""
    setup_dirs()

    # 1. Dataset loading & Preprocessing
    X, y, classes, class_counts, sample_images = load_and_preprocess()

    # 2. Preprocessing & Dataset distribution visualizations
    print("\nSTEP 2: GENERATING DATASET & PREPROCESSING VISUALIZATIONS")
    plot_preprocessing(sample_images, classes)
    plot_dataset_dist(class_counts)

    # 3. Stratified Dataset Splitting: 70% Train, 15% Val, 15% Test
    print("\nSTEP 3: SPLITTING DATASET (70% Train, 15% Validation, 15% Test)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"  • Training Samples   : {X_train.shape[0]}")
    print(f"  • Validation Samples : {X_val.shape[0]}")
    print(f"  • Test Samples       : {X_test.shape[0]}")

    # 4. Model Definition
    print("\nSTEP 4: BUILDING CONVOLUTIONAL NEURAL NETWORK")
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(classes))
    model.summary()

    # 5. Training
    print(f"\nSTEP 5: TRAINING CNN MODEL ({epochs} EPOCHS)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 6. Evaluation on Unseen Test Data
    print("\nSTEP 6: EVALUATING CNN ON UNSEEN TEST SET")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n>> Final Test Accuracy : {test_acc * 100:.2f}%")
    print(f">> Final Test Loss     : {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)

    print("\n" + "="*65)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*65)
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    print("="*65)

    # Save Plots
    plot_training(history)
    plot_confusion_matrix(cm, classes)

    # Save Model
    print(f"\nSaving model to: {MODEL_PATH}")
    model.save(MODEL_PATH)

    # Save Metrics JSON
    final_train_acc = float(history.history['accuracy'][-1])
    final_val_acc = float(history.history['val_accuracy'][-1])
    final_train_loss = float(history.history['loss'][-1])
    final_val_loss = float(history.history['val_loss'][-1])

    metrics = {
        "classes": classes,
        "class_counts": class_counts,
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "image_size": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "epochs_trained": int(epochs),
        "train_accuracy": round(final_train_acc * 100, 2),
        "val_accuracy": round(final_val_acc * 100, 2),
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "test_accuracy": round(float(test_acc) * 100, 2),
        "test_loss": round(float(test_loss), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {METRICS_PATH}")

    print("\n" + "#"*65)
    print("ALL TRAINING & EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print("#"*65 + "\n")
    return model, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN Model for Blood Cell Image Classification")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument('--force', action='store_true', help="Force retraining")
    args = parser.parse_args()

    train_and_evaluate(epochs=args.epochs, force=args.force)
