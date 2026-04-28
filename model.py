import torch.nn as nn
from torchvision import models

class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return features, embeddings