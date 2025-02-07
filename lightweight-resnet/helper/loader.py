import torch
import torch.nn as nn

class Loader():
    def __init__(self, device = None):
        self.device = device
        
    def model_load(self, file_name : str):
        model = torch.load(file_name, map_location = self.device)    
        model.eval()
        print("===Model loaded!===")
        return model