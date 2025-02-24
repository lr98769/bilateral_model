from torch.nn import Module, Linear
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class ResNet18(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace layer to suit number of classes
        self.model.fc = Linear(512, self.num_classes)

    def forward(self, x):
        return self.model(x)
