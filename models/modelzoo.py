import os

from models.efficientnet.model import EfficientNetModel
from models.resnet50.models import ResNet50Model
from models.swin_transformer.model import SwinTransformerModel


def get_model(model_name, **model_config):
    model_name_lower = model_name.lower()
    
    if model_name_lower == "resnet50":
        return ResNet50Model(**model_config['args'])
    elif model_name_lower == "efficientnet":
        return EfficientNetModel(**model_config['args'])
    elif model_name_lower == "swin_transformer":
        return SwinTransformerModel(**model_config['args'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

