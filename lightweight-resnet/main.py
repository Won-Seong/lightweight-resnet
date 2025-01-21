import torch
from models.resnet50 import ResNet50
from helper.data_generator import cifar100
from helper.evaluator import Evaluator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    
    model = ResNet50(device)
    #train_data_loader = cifar100(path = './datasets/train')
    test_data_loader = cifar100(path = './datasets/test')
    evaluator = Evaluator(device)
        
    evaluator.evaluate(model, test_data_loader)