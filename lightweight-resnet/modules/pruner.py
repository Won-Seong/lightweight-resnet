import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.fx import symbolic_trace
from copy import deepcopy

from helper.trainer import Trainer
from helper.evaluator import number_of_parameters
import pandas as pd

from modules.distillation_loss import DistillationLoss

class Pruner():
    def __init__(self, model : nn.Module):
        self.model = symbolic_trace(model)
        self.trainer = Trainer(self.model, DistillationLoss())
        self.teacher = deepcopy(self.model)
        
    def iterative_prune(self, num_of_iter : int, train_data_loader : DataLoader, test_data_loader : DataLoader,
                        epochs : int, n : int = 2, amount=0.1):
        logs = []
        for i in range(1, num_of_iter + 1):
            self.trainer = Trainer(self.model, DistillationLoss())
            self.prune(n, amount)
            print("Step = " + str(i) + " | Num of parameters = " + str(number_of_parameters(self.model)))
            best_val_accuracy = self.re_train(train_data_loader, test_data_loader, epochs, 
                                              'check_points/prune_iter_' + str(i), 'runs/prune_iter_' + str(i))
            logs.append({"Step" : i, "Parameters" : number_of_parameters(self.model), "Accuracy" : best_val_accuracy})
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv("pruning_logs.csv", index=False)
        
    def prune(self, n=2, amount=0.1):
        modules_to_prune = []
        for name, module in self.model.named_modules():
            if 'layer' not in name or 'downsample' in name:
                continue
            elif isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
                modules_to_prune.append((name, module))

        for name, module in modules_to_prune:
            if 'conv1' in name: # First layer
                prune.ln_structured(module, name='weight', amount = amount, n = n, dim = 0)
                prune.remove(module, 'weight')
                zero_idx = []
                for idx in range(len(module.weight.data)):
                    if torch.all(module.weight.data[idx] == 0):
                        zero_idx.append(idx)
                all_idx = torch.arange(module.weight.data.size(0))
                mask_idx = ~torch.isin(all_idx, torch.tensor(zero_idx))
                pruned_weight = module.weight[mask_idx]
                new_module = nn.Conv2d(module.in_channels, pruned_weight.size(0), kernel_size=module.kernel_size,
                                    stride=module.stride, padding=module.padding)
                new_module.weight.data = pruned_weight
            elif 'conv3' in name:
                pruned_weight = module.weight[:, mask_idx, :, :]
                new_module = nn.Conv2d(pruned_weight.size(1), module.out_channels, kernel_size=module.kernel_size,
                                stride=module.stride, padding=module.padding)
                new_module.weight.data = pruned_weight
                mask_idx = [True] * new_module.out_channels
            elif isinstance(module, torch.nn.Conv2d):
                pruned_weight = module.weight[:, mask_idx, :, :]
                new_module = nn.Conv2d(pruned_weight.size(1), module.out_channels, kernel_size=module.kernel_size,
                                stride=module.stride, padding=module.padding)
                new_module.weight.data = pruned_weight
                prune.ln_structured(new_module, name='weight', amount = amount, n = 2, dim = 0)
                prune.remove(new_module, 'weight')
                zero_idx = []
                for idx in range(len(new_module.weight.data)):
                    if torch.all(new_module.weight.data[idx] == 0):
                        zero_idx.append(idx)
                all_idx = torch.arange(new_module.weight.data.size(0))
                mask_idx = ~torch.isin(all_idx, torch.tensor(zero_idx))
                pruned_weight = new_module.weight[mask_idx]
                new_module = nn.Conv2d(new_module.in_channels, pruned_weight.size(0), kernel_size=module.kernel_size,
                                    stride=module.stride, padding=module.padding)
                new_module.weight.data = pruned_weight
            elif isinstance(module, torch.nn.BatchNorm2d):
                pruned_weight = module.weight[mask_idx]
                pruned_bias = module.bias[mask_idx]
                pruned_running_mean = module.running_mean[mask_idx]
                pruned_running_var = module.running_var[mask_idx]
                new_module = nn.BatchNorm2d(pruned_weight.size(0))
                new_module.weight.data = pruned_weight
                new_module.bias.data = pruned_bias
                new_module.running_mean.data = pruned_running_mean
                new_module.running_var.data = pruned_running_var
            self.model.add_submodule(name, new_module)
            
    def re_train(self, train_data_loader : DataLoader, test_data_loader : DataLoader,
                    epochs : int, file_name : str, log_dir : str):
        self.trainer.distillation_train(self.teacher, train_data_loader, test_data_loader, epochs, file_name, log_dir)
        return self.trainer.best_val_accuracy
        
            