import numpy as np
class ImageLabelConverter:
    def __init__(self, num_classes):
        # Assume higher class number = more severe
        self.num_classes = num_classes
        
    def encode_class_num(self, class_num: int) -> np.array:
        # 2. Encode into Special Ordinal Style
        # - 4 is the worst state, 0 is the least severe state
        # if LE is 4 and RE is 0, we should output 4
        # 4 = [1 0 0 0, 0], 0 = [1, 1, 1, 1, 1], 4*1 = [1 0 0 0, 0]
        num_ones = self.num_classes-class_num
        return np.concatenate((np.repeat(1, num_ones), np.repeat(0, class_num)))
    
    def decode_label(self, label: np.array) -> int:
        # Assume that label is already sigmoid
        # [1 0 0 0, 0] -> 4
        return int(self.num_classes-sum(label.round()))
    
    def decode_labels(self, label: np.array) -> int:
        # Assume that label is already sigmoid
        # [1 0 0 0, 0] -> 4
        return self.num_classes-label.round().sum(axis=1)
        