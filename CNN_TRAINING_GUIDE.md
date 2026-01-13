# CNN Training Guide for Pechay Leaf Detection

## Overview
This guide will help you train a CNN model to detect pechay leaf conditions: Healthy, Diseased, Pest Damage, and Nutrient Deficiency.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Create the following directory structure:
```
pechay_dataset/
├── train/
│   ├── Healthy/
│   │   ├── healthy_leaf_001.jpg
│   │   ├── healthy_leaf_002.jpg
│   │   └── ...
│   └── Diseased/
│       ├── diseased_leaf_001.jpg
│       └── ...
├── val/
│   ├── Healthy/
│   └── Diseased/
└── test/
    ├── Healthy/
    └── Diseased/
```

### 3. Dataset Requirements
- **Minimum images per class**: 50-100 for training, 10-20 for validation/test
- **Image formats**: JPG, PNG, JPEG
- **Image size**: Any size (will be resized to 224x224)
- **Quality**: Clear, well-lit images of pechay leaves
- **Classes**: Only 2 classes - Healthy and Diseased

## Training Process

### 1. Run Training Script
```bash
python train_cnn.py
```

The script will:
- Create dataset directory structure if it doesn't exist
- Load and preprocess images with data augmentation
- Train a ResNet18-based CNN model
- Save the best model as `pechay_cnn_model_YYYYMMDD_HHMMSS.pth`
- Generate training plots and confusion matrix

### 2. Training Parameters
You can modify these in `train_cnn.py`:
- `batch_size`: 32 (adjust based on GPU memory)
- `num_epochs`: 50 (increase for better accuracy)
- `learning_rate`: 0.001 (lower for fine-tuning)
- `data_dir`: "pechay_dataset" (your dataset path)

### 3. Monitor Training
The script will display:
- Training/validation loss and accuracy per epoch
- Best validation accuracy
- Training progress

## Model Architecture

The CNN uses:
- **Backbone**: Pre-trained ResNet18 (transfer learning)
- **Frozen layers**: Early layers frozen, only final layers trained
- **Final layers**: Dropout + Linear(128) + ReLU + Dropout + Linear(2)
- **Classes**: 2 (Healthy, Diseased)

## Data Augmentation

Training images are augmented with:
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- Random translation (10% of image size)
- Normalization (ImageNet mean/std)

## Evaluation

After training, the model will:
- Generate a confusion matrix (`confusion_matrix.png`)
- Show classification report with precision/recall
- Plot training history (`training_history.png`)

## Integration with Flask App

### 1. Copy Model File
After training, copy your best model:
```bash
copy pechay_cnn_model_YYYYMMDD_HHMMSS.pth pechay_cnn_model.pth
```

### 2. Test Prediction
```bash
python predict.py
```

### 3. Run Flask App
```bash
python app.py
```

The Flask app will automatically:
- Load the CNN model if available
- Use CNN predictions for uploaded images
- Fall back to simulation if no model is found

## Tips for Better Results

### 1. Data Collection
- Collect images in various lighting conditions
- Include different angles and distances
- Ensure balanced dataset across all classes
- Use high-quality, clear images

### 2. Training Tips
- Start with more epochs (100+) for better accuracy
- Use GPU if available (CUDA)
- Monitor for overfitting (validation accuracy plateauing)
- Try different learning rates (0.0001, 0.01)

### 3. Model Improvements
- Collect more data for underrepresented classes
- Try different architectures (ResNet50, EfficientNet)
- Use ensemble methods
- Fine-tune hyperparameters

## Troubleshooting

### Common Issues
1. **"No training data found"**: Check dataset directory structure
2. **CUDA out of memory**: Reduce batch_size
3. **Poor accuracy**: Collect more data or increase training epochs
4. **Model not loading**: Ensure model file exists and is compatible

### Performance Expectations
- **Good accuracy**: 85-95% on validation set
- **Training time**: 30-60 minutes on GPU, 2-4 hours on CPU
- **Model size**: ~45MB (ResNet18)

## Next Steps

1. **Collect more data** for better accuracy
2. **Experiment with architectures** (ResNet50, EfficientNet)
3. **Implement real-time detection** with camera feed
4. **Add more classes** (specific diseases, pest types)
5. **Deploy to production** with proper error handling

## Files Created

After training, you'll have:
- `pechay_cnn_model_YYYYMMDD_HHMMSS.pth`: Trained model
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training/validation curves
- `pechay_dataset/`: Your organized dataset

## Support

If you encounter issues:
1. Check dataset structure matches requirements
2. Verify all dependencies are installed
3. Ensure sufficient disk space for model files
4. Check GPU memory if using CUDA
