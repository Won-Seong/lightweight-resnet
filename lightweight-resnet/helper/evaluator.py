import torch
from tqdm import tqdm
import time

@torch.no_grad()    
def number_of_parameters(model, print_result=False):
    total_params = sum(p.numel() for p in model.parameters())
    if print_result:
        print(f"Total number of parameters: {total_params}")
    return total_params

class Evaluator():
    def __init__(self, device):
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, model, data_loader, print_result=False):
        model.eval()
        accuracy = 0
        latency = 0
        
        # warm-up
        for _ in range(3):
            batch = next(iter(data_loader))
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            outputs = model(x)
    
        for batch in tqdm(data_loader):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(x)
            torch.cuda.synchronize()
            end_time = time.time()
            y_hat = outputs.argmax(dim=1)
            latency += (end_time - start_time)
            accuracy += (y_hat == y).sum().item()
    
        accuracy /= len(data_loader.dataset)
        latency /= len(data_loader)
        if print_result:
            print(f"Average latency per batch: {latency * 1000:.2f} ms")
            print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy