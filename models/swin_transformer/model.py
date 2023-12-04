import timm


class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.base_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
