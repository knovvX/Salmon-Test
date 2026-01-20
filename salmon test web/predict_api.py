"""
Flask API server for fish origin classification
Serves the predict_web.html frontend and provides prediction API

SECURITY NOTES:
- Images are processed in memory only (no disk storage)
- Default: localhost-only access (127.0.0.1)
- CORS restricted to same-origin requests
- Request size limited to 50MB
- Generic error messages returned to clients
- Debug mode disabled by default

USAGE:
  Development:  FLASK_DEBUG=true python predict_api.py
  Production:   python predict_api.py
  Custom host:  FLASK_HOST=0.0.0.0 python predict_api.py
"""

import sys
import os
from pathlib import Path
import base64
import io

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import tifffile as tiff

from src.config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.model.simple_multimodal_cnn import create_simple_multimodal_cnn
from src.utils import get_device
from src.image_preprocessing import preprocess_img_pipeline

app = Flask(__name__, static_folder='.')
# Security: Limit request size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# CORS configuration - restrict to localhost for security
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5001", "http://127.0.0.1:5001", "http://localhost:8501", "http://127.0.0.1:8501"],
        "methods": ["POST", "GET"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables
device = None
model = None
class_names = ['Hatchery', 'Natural']

# FL normalization parameters
FL_APPROX_MEAN = 60.0
FL_APPROX_STD = 10.0

def normalize_fl(fl_value):
    """Normalize FL value"""
    return (fl_value - FL_APPROX_MEAN) / FL_APPROX_STD

def preprocess_image(img_array):
    """Preprocess image using the same pipeline as training"""
    # Core preprocessing pipeline
    img_proc = preprocess_img_pipeline(img_array)  # H W 3, [0,1]
    
    # Convert to PIL RGB for torchvision transforms
    pil_img = Image.fromarray((img_proc * 255).astype(np.uint8), mode="RGB")
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    img_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def grad_cam(model, image, tabular_features, device, target_class=None):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Find the last convolutional layer in image_features
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = name
    
    if target_layer is None:
        return None, None
    
    activations = {}
    gradients = {}
    
    def forward_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if name == target_layer:
            handle_forward = module.register_forward_hook(forward_hook(name))
            handle_backward = module.register_full_backward_hook(backward_hook(name))
            handles = [handle_forward, handle_backward]
            break
    
    # Forward pass
    image = image.to(device)
    image.requires_grad = True
    
    output = model(image, tabular_features)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Generate CAM
    if target_layer in gradients and target_layer in activations:
        grad = gradients[target_layer]
        act = activations[target_layer]
        
        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return cam, target_class
    
    # Remove hooks if failed
    for handle in handles:
        handle.remove()
    
    return None, None

def image_to_base64(img_array):
    """Convert image array to base64 PNG string"""
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def overlay_heatmap(img_array, heatmap):
    """Overlay heatmap on image"""
    h, w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay with alpha blending
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

@app.route('/')
def index():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'predict_web.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        
        # Get image data
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get tabular features
        sex = int(data.get('sex', 2))  # Default: Unknown
        fl = float(data.get('fl', 60))  # Default: 60cm
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        img_array = tiff.imread(io.BytesIO(image_bytes))
        
        # Convert to RGB if grayscale
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Normalize to uint8 if needed
        if img_array.dtype != np.uint8:
            if img_array.max() > 1.0:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = (img_array * 255).astype(np.uint8)
        
        # Preprocess image
        img_tensor = preprocess_image(img_array)
        
        # Prepare tabular features
        fl_value = normalize_fl(fl)
        
        tabular_features = {
            'sex': torch.tensor([sex], dtype=torch.long).to(device),
            'fl': torch.tensor([fl_value], dtype=torch.float32).to(device),
            'year': torch.tensor([0], dtype=torch.long).to(device)
        }
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor.to(device), tabular_features)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Generate Grad-CAM
        heatmap, _ = grad_cam(model, img_tensor, tabular_features, device)
        
        # Prepare response
        result = {
            'prediction': prediction,
            'class_name': class_names[prediction],
            'confidence': float(confidence),
            'probabilities': [
                float(probabilities[0, 0].item()),
                float(probabilities[0, 1].item())
            ],
            'original_image': image_to_base64(img_array),
            'heatmap_image': image_to_base64(overlay_heatmap(img_array, heatmap)) if heatmap is not None else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        # Log to server console for debugging (not visible to client)
        print(f"[ERROR] Prediction failed: {str(e)}")
        traceback.print_exc()
        # Return generic error message to client (don't expose sensitive info)
        return jsonify({'error': 'Prediction failed. Please check your image format and try again.'}), 500

def load_model():
    """Load the trained model"""
    global device, model
    
    print("Loading model...")
    device = get_device()
    
    # Path to the latest best checkpoint from recent training
    checkpoint_path = os.path.join(project_root, "best.ckpt")
    if not os.path.exists(checkpoint_path):
        # Fallback to older location if needed
        checkpoint_path = os.path.join(project_root, "results/simple_multimodal_cnn_20251201_214610/checkpoints/best.ckpt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model matching the latest 6-layer architecture with adaptive pooling
    # dropout_rate=0.4 matches the latest training configuration
    model = create_simple_multimodal_cnn(num_classes=2, dropout_rate=0.4, use_year=False, num_layers=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"✅ Device: {device}")

if __name__ == '__main__':
    print("=" * 60)
    print("Fish Origin Classification API Server")
    print("=" * 60)
    
    # Load model
    load_model()
    
    # Security settings
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '127.0.0.1')  # Default to localhost only
    port = int(os.getenv('FLASK_PORT', '5001'))  # Changed to 5001 to avoid macOS AirPlay conflict
    
    if debug_mode:
        print("⚠️  WARNING: Debug mode is enabled! DO NOT use in production!")
    
    print(f"\n🚀 Starting Flask server...")
    print(f"📱 Access: http://{host}:{port}")
    print(f"🔒 Security: Debug={debug_mode}, Host={host}")
    print("=" * 60)
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug_mode)

