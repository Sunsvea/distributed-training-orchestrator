import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class SimpleCNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        
        # Calculate conv output size
        # Assuming input is flattened from 28x28 (MNIST-like)
        self.input_channels = 1
        self.conv_size = int(input_size ** 0.5)  # Assuming square input
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size after conv layers
        conv_output_size = (self.conv_size // 4) ** 2 * 64
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # Reshape input to image format if needed
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, self.input_channels, self.conv_size, self.conv_size)
        
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SimpleFC(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SimpleFC, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_heads: int = 8, num_layers: int = 2):
        super(SimpleTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding (simple learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Project to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

def create_model(model_config: Dict[str, Any]) -> Optional[nn.Module]:
    model_type = model_config.get("model_type", "simple_fc")
    input_size = model_config.get("input_size", 784)
    hidden_size = model_config.get("hidden_size", 128)
    num_classes = model_config.get("num_classes", 10)
    
    try:
        if model_type == "simple_cnn":
            return SimpleCNN(input_size, hidden_size, num_classes)
        elif model_type == "simple_fc":
            return SimpleFC(input_size, hidden_size, num_classes)
        elif model_type == "simple_transformer":
            num_heads = model_config.get("num_heads", 8)
            num_layers = model_config.get("num_layers", 2)
            return SimpleTransformer(input_size, hidden_size, num_classes, num_heads, num_layers)
        else:
            print(f"Unknown model type: {model_type}")
            return None
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    if model is None:
        return {}
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "layer_names": [name for name, _ in model.named_parameters()],
        "layer_shapes": {name: list(param.shape) for name, param in model.named_parameters()}
    }

def count_model_parameters(model: nn.Module) -> int:
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)