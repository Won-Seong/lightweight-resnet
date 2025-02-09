# **Lightweight ResNet50 Project**

This repository focuses on compressing **ResNet50** to create a **lightweight-ResNet50** while maintaining competitive performance. The compression process involves iterative **pruning** and **knowledge distillation** with the following setup:

## **Compression Strategy**
- The dataset used is **CIFAR-100**, and the model is initialized with a **pre-trained ResNet50 on ImageNet**.
- **Pruning and knowledge distillation** are applied iteratively.
- **Pruning is performed on convolutional layers and batch normalization layers**, excluding the residual connections.
- At each step, **the number of parameters in each layer is reduced by 10%**.
- After pruning, **the model is retrained for 30 epochs** with a learning rate of **1e-4**.

## **Results**

The table below summarizes the results of compressing ResNet50. The full results can be found in [this file](assets/pruning_logs.csv).

| Number of Parameters | Validation Accuracy | Size (MB) |
|----------------------|---------------------|-----------|
| 23,712,932 | 82.50% | 95.19 |
| 20,639,037 | 82.87% | 79.08 |
| 18,051,107 | 82.25% | 69.21 |
| 12,485,140 | 79.83% | 47.96 |
| 10,078,784 | 77.45% | 38.77 |
| 7,663,862 | 74.05% | 29.55 |
| 6,146,234 | 69.94% | 23.76 |
| 5,154,387 | 65.28% | 19.97 |
| 4,336,346 | 60.30% | 16.85 |

### **Key Observations**
- **In the first pruning step, over 3 million parameters were removed, yet accuracy slightly increased.**
- **When approximately 10 million parameters were removed, accuracy dropped to the 70% range.**
- **At around 17 million parameters removed (~74% reduction), accuracy fell to the 60% range.**

## **Accuracy vs. Number of Parameters**

The following plot visualizes the relationship between **the number of parameters and accuracy**.

<p align="center">
  <img src="assets/accu_over_params.png" width="800"/>
</p>

### **Insights from the Plot**
- **Accuracy gradually decreases as the number of parameters decreases.**
- **Initially, the performance degradation is minor, but after a certain threshold, accuracy drops sharply.**
- **Excessive pruning leads to significant accuracy loss, making it crucial to determine an optimal pruning threshold.**

## **Can the Performance Be Improved?**

Yes! Performance can be further improved.  
For instance, **after retraining the model with 7,663,862 parameters, accuracy improved from 74.05% to 74.57%**.  
This suggests that applying **longer retraining schedules, data augmentation, and fine-tuning strategies** can yield better results.

## **How to Use This Repository**

The following script **reproduces the experiments**.  
Pruned models are saved in the `check_points` folder, and the pruning logs are recorded in `pruning_logs.csv`.

```python
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
    print(f'Using device: {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU'))
    
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
```

## **Suggestions for Improvement**
1. **Hyperparameter Optimization**  
   - Try adjusting **pruning percentages** for different layers instead of a fixed 10% reduction.  
   - Experiment with different **learning rates, batch sizes, and optimizers** for retraining.  

2. **More Advanced Knowledge Distillation**  
   - Implement **attention-based distillation** to preserve important feature representations.  
   - Use **multiple teacher models** to distill knowledge from an ensemble of networks.

3. **Visual Analysis**  
   - Plot **layer-wise pruning effects** to analyze which layers contribute the most to accuracy drops.  
   - Use **t-SNE** or **PCA** to visualize how feature representations change before and after pruning.

## **Final Thoughts**
This project demonstrates that **ResNet50 can be significantly compressed while maintaining reasonable accuracy**.  
However, careful fine-tuning and training strategies are required to **mitigate accuracy loss from excessive pruning**.  

If you have any questions or suggestions, feel free to contribute!

