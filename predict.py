import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path

class PechayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PechayCNN, self).__init__()
        
        # Use pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def get_features(self, x):
        """Extract features before the final classification layer"""
        # Hook into the backbone to get features
        # ResNet18 structure: ... -> avgpool -> fc
        # We need to bypass the modified fc layer or extract from it
        
        # Method 1: Forward pass until avgpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class PechayPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.class_names = ['Healthy', 'Diseased']
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = PechayCNN(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {self.class_names}")
    
    def extract_features(self, image_path):
        """Extract feature vector from image"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.get_features(input_tensor)
                return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict_image(self, image_path):
        """Predict leaf condition from image path"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted.item()]
                confidence_score = confidence.item() * 100
                
                # Get all class probabilities
                all_probs = probabilities.squeeze().cpu().numpy()
                # Convert NumPy float32 to native Python float for JSON serialization
                class_probs = {self.class_names[i]: float(prob * 100) for i, prob in enumerate(all_probs)}
                
                # Check for "Not Pechay" based on confidence threshold
                # If the highest confidence is too low, it's likely not a pechay or a very confusing image
                CONFIDENCE_THRESHOLD = 75.0  # Threshold for "Not Pechay"
                
                if confidence_score < CONFIDENCE_THRESHOLD:
                    predicted_class = "Not Pechay"
                    # Keep the confidence score as is, but label it as such
                
                return {
                    'condition': predicted_class,
                    'confidence': round(confidence_score, 2),
                    'all_probabilities': class_probs,
                    'success': True
                }
                
        except Exception as e:
            return {
                'condition': 'Error',
                'confidence': 0,
                'all_probabilities': {},
                'success': False,
                'error': str(e)
            }
    
    def predict_from_bytes(self, image_bytes):
        """Predict leaf condition from image bytes"""
        try:
            from io import BytesIO
            
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted.item()]
                confidence_score = confidence.item() * 100
                
                # Get all class probabilities
                all_probs = probabilities.squeeze().cpu().numpy()
                # Convert NumPy float32 to native Python float for JSON serialization
                class_probs = {self.class_names[i]: float(prob * 100) for i, prob in enumerate(all_probs)}
                
                return {
                    'condition': predicted_class,
                    'confidence': round(confidence_score, 2),
                    'all_probabilities': class_probs,
                    'success': True
                }
                
        except Exception as e:
            return {
                'condition': 'Error',
                'confidence': 0,
                'all_probabilities': {},
                'success': False,
                'error': str(e)
            }

def main():
    """Test the predictor with a sample image"""
    # Example usage
    model_path = "pechay_cnn_model_20251212_184656.pth"  # Change to your trained model path
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please train a model first using train_cnn.py")
        return
    
    # Initialize predictor
    predictor = PechayPredictor(model_path)
    
    # Test with sample image (replace with your image path)
    sample_image = "sample_leaf.jpg"
    
    if os.path.exists(sample_image):
        result = predictor.predict_image(sample_image)
        
        if result['success']:
            print(f"Predicted Condition: {result['condition']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print("\nAll Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.2f}%")
        else:
            print(f"Prediction failed: {result['error']}")
    else:
        print(f"Sample image {sample_image} not found!")
        print("Please provide a valid image path to test the predictor.")

if __name__ == "__main__":
    main()
