"""
Simple Multimodal CNN - combines Simple CNN with tabular features
Best of both worlds: Simple architecture + multimodal information
"""

import torch
import torch.nn as nn


class SimpleMultimodalCNN(nn.Module):
    """
    Multimodal CNN combining image and tabular features
    
    Architecture:
        Image → CNN (4 or 6 layers) → Adaptive Pool → 128-dim features
        Sex, FL → Embedding → 4-dim features
        Combined → Fusion MLP → Classification
    
    ✨ Supports flexible input sizes (224x224, 384x384, 512x512, 1024x1024, etc.)
       Uses AdaptiveAvgPool2d to normalize feature maps to 8x8 regardless of input size
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, use_year=False, num_layers=4):
        super(SimpleMultimodalCNN, self).__init__()
        
        self.use_year = use_year
        self.num_layers = num_layers
        
        # ========== Image Branch (Simple CNN) ==========
        
        if num_layers == 4:
            # ===== 4-LAYER CNN =====
            # More compact for smaller datasets or simpler features (like FFT)
            # Reduces overfitting risk on FFT magnitude spectra
            self.image_features = nn.Sequential(
                # Block 1: 3 -> 32
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/2 x W/2
                
                # Block 2: 32 -> 64
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/4 x W/4
                
                # Block 3: 64 -> 128
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/8 x W/8
                
                # Block 4: 128 -> 128
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/16 x W/16
            )
        else:
            # ===== 6-LAYER CNN (VGG-style) =====
            # Deeper network for better feature extraction from high-res fish scale images
            self.image_features = nn.Sequential(
                # Block 1: 2 Conv layers
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/2 x W/2
                
                # Block 2: 2 Conv layers
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/4 x W/4
                
                # Block 3: 2 Conv layers
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # H/8 x W/8
                nn.MaxPool2d(2, 2),  # H/16 x W/16
            )
        
        # Adaptive pooling to fixed size (8x8) - works with ANY input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Image feature dimension after adaptive pooling: 128 * 8 * 8 = 8192 (FIXED)
        self.image_feature_dim = 128 * 8 * 8
        
        # Image feature projection to lower dimension
        self.image_projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.image_feature_dim, 128),
            nn.ReLU(inplace=True)
        )
        
        # ========== Tabular Branch ==========
        # Sex embedding: 3 classes (Female=0, Male=1, Unknown=2) -> 3 dims
        self.sex_embedding = nn.Embedding(3, 3)
        
        # Year embedding: up to 10 years -> 4 dims
        if use_year:
            self.year_embedding = nn.Embedding(10, 4)
            self.tabular_feature_dim = 3 + 4 + 1  # sex_emb + year_emb + fl = 8
        else:
            self.tabular_feature_dim = 3 + 1  # sex_emb + fl = 4
        
        # ========== Fusion Branch ==========
        combined_dim = 128 + self.tabular_feature_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Store config
        self.config = {
            'image_feature_dim': 128,
            'tabular_feature_dim': self.tabular_feature_dim,
            'combined_dim': combined_dim,
            'dropout_rate': dropout_rate,
            'num_classes': num_classes,
            'num_layers': num_layers
        }
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, image, tabular_features):
        """
        Forward pass
        """
        # ========== Image Features ==========
        img_features = self.image_features(image)
        img_features = self.adaptive_pool(img_features)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.image_projection(img_features)
        
        # ========== Tabular Features ==========
        sex_emb = self.sex_embedding(tabular_features['sex'])
        fl_norm = tabular_features['fl'].unsqueeze(1)
        
        if self.use_year:
            year_emb = self.year_embedding(tabular_features['year'])
            tab_features = torch.cat([sex_emb, year_emb, fl_norm], dim=1)
        else:
            tab_features = torch.cat([sex_emb, fl_norm], dim=1)
        
        # ========== Fusion ==========
        combined = torch.cat([img_features, tab_features], dim=1)
        output = self.fusion(combined)
        
        return output
    
    def print_model_info(self):
        """Print model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "=" * 70)
        print(f"{self.num_layers}-LAYER MULTIMODAL CNN MODEL INFO")
        print("=" * 70)
        print(f"✨ Architecture: {self.num_layers} convolutional layers")
        print(f"✨ Flexible input size: ANY resolution")
        print(f"Image feature dim: {self.config['image_feature_dim']}")
        print(f"Tabular feature dim: {self.config['tabular_feature_dim']}")
        print(f"Combined feature dim: {self.config['combined_dim']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        print("=" * 70 + "\n")


def create_simple_multimodal_cnn(num_classes=2, dropout_rate=0.5, use_year=False, num_layers=4):
    """
    Create SimpleMultimodalCNN model
    """
    return SimpleMultimodalCNN(num_classes=num_classes, dropout_rate=dropout_rate, use_year=use_year, num_layers=num_layers)


if __name__ == "__main__":
    # Test 4-layer model
    print("Testing 4-layer model:")
    model4 = create_simple_multimodal_cnn(num_layers=4)
    model4.print_model_info()
    
    # Test 6-layer model
    print("Testing 6-layer model:")
    model6 = create_simple_multimodal_cnn(num_layers=6)
    model6.print_model_info()
