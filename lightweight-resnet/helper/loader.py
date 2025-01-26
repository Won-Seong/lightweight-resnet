import torch
import torch.nn as nn

class Loader():
    def __init__(self, device = None):
        self.device = device
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Best validation accuracy: " + str(check_point["val_accuracy"]))
        
    def model_load(self, file_name : str, model : nn.Module, 
             print_dict : bool = True):
        check_point = torch.load(file_name + ".pth", map_location=self.device)
        if print_dict: self.print_model(check_point)
        model.load_state_dict(check_point["model_state_dict"])
        model.eval()
        print("===Model loaded!===")
        return model
        
    def load_for_training(self, file_name : str, model : nn.Module):
        check_point = torch.load(file_name + ".pth", map_location=self.device)
        self.print_model(check_point)
        model.load_state_dict(check_point["model_state_dict"])
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        epoch = check_point["epoch"]
        best_val_accuracy = check_point["val_accuracy"]
        print("===Model/Optimizer/Epoch/Accuracy loaded!===")
        return model, optimizer, epoch, best_val_accuracy