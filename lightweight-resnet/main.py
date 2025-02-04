import torch
from helper.trainer import Trainer
from helper.loader import Loader
from models.resnet50 import get_resnet50
from helper.data_generator import cifar100
from helper.evaluator import Evaluator
from modules.distillation_loss import DistillationLoss
from modules.pruner import Pruner

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    loader = Loader(device)
    evaluator = Evaluator(device)
    
    model = get_resnet50(100).to(device)
    loader.model_load('check_points/original', model)
    
    train_data_loader = cifar100(path = './datasets/train', batch_size=256)
    test_data_loader = cifar100(path = './datasets/test', batch_size=256)
    
    pruner = Pruner(model, train_data_loader)
    pruner.iterative_prune(10, test_data_loader, 20)
    
    #evaluator.evaluate(model, test_data_loader, True)
    #trainer = Trainer(model, torch.nn.CrossEntropyLoss())
    #trainer.train(train_data_loader, test_data_loader, 300, 'check_points/original', 'runs/original')
    
    