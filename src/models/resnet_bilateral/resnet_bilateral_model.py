from torch.nn import Module, Linear, Sequential
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch

from src.models.resnet_bilateral.resnet_decoder import Decoder, BasicBlockDec

class ResNet18Bilateral(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = Sequential()
        for name, child in list(resnet.named_children())[:-2]:
            self.encoder.add_module(name, child)
        self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2]) 
        self.pooling = list(resnet.children())[-2]
        self.fc = Linear(512, self.num_classes)
    
    def _class_predict(self, encoded):
        pooled = self.pooling(encoded)
        pooled = torch.flatten(pooled, start_dim=1)
        logits = self.fc(pooled)
        prediction = torch.sigmoid(logits)
        return prediction
    
    def reconstruction(self, eye):
        encoded = self.encoder(eye)
        decoded = self.decoder(encoded)
        return decoded
        
    def eye_prediction(self, eye):
        encoded = self.encoder(eye)
        prediction = self._class_predict(encoded)
        return prediction

    def patient_prediction(self, left, right):
        left_pred = self.eye_prediction(left)
        right_pred = self.eye_prediction(right)
        return left_pred*right_pred
        
    def forward(self, left, right):
        return self.patient_prediction(left, right)
    

    
