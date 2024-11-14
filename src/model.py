import torch
import torch.nn as nn
from torchvision import models
from src.config import Config
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def create_model():
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(Config.DEVICE)

