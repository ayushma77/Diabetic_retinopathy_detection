from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
