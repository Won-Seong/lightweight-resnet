import torch
from helper.trainer import Trainer
from helper.loader import Loader
from models.resnet50 import get_resnet50
from helper.data_generator import cifar100
from helper.evaluator import Evaluator, number_of_parameters
from modules.distillation_loss import DistillationLoss
from modules.pruner import Pruner

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    loader = Loader(device)
    evaluator = Evaluator(device)
    
    # Load dataset
    train_data_loader = cifar100(path='./datasets/train', batch_size=256, train=True)
    test_data_loader = cifar100(path='./datasets/test', batch_size=256, train=False)
    
    # Load a pre-trained model on ImageNet and fine-tune it
    original = get_resnet50(100)
    trainer = Trainer(original, torch.nn.CrossEntropyLoss())
    trainer.train(train_data_loader, test_data_loader, 100, 'check_points/original', 'runs/original')
    
    # Load the teacher model for Knowledge Distillation
    teacher = loader.model_load('check_points/original')
    model = loader.model_load('check_points/original')
    
    # Perform iterative pruning followed by Knowledge Distillation retraining
    pruner = Pruner(teacher, model)
    pruner.iterative_prune(20, train_data_loader, test_data_loader, 30)
    
    # Measure the number of parameters and accuracy after all processes
    number_of_parameters(model, print_result=True)    
    evaluator.evaluate(model, test_data_loader, True)
    
    # Retrain the pruned model if necessary
    trainer = Trainer(model, DistillationLoss(5.0, 0.7))
    trainer.distillation_train(teacher, train_data_loader, test_data_loader, 50, 'check_points/after_pruning', 'runs/after_pruning')