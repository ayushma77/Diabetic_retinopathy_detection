import torch.nn as nn
import torchvision.models as models


class ResNet50Model(nn.Module):
    def __init__(self, num_classes,dropout_prob=0.5):
        super(ResNet50Model, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)
