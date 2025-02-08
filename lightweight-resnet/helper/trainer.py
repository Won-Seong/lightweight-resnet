import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from helper.evaluator import Evaluator

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer = None,
                 start_epoch = 0,
                 best_val_accuracy = 0.0,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.98)
        self.accelerator = Accelerator(mixed_precision = 'no')
        self.start_epoch = start_epoch
        self.best_val_accuracy = best_val_accuracy
        self.evaluator = Evaluator(self.accelerator.device)

    def train(self, train_data_loader : DataLoader, test_data_loader : DataLoader, epochs : int, file_name : str, log_dir : str):
        writer = SummaryWriter(log_dir = log_dir)
        self.model.train()
        self.model, self.optimizer, train_data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_data_loader, self.scheduler)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                self.accelerator.backward(loss)
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
            
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.scheduler.step()
                epoch_loss = epoch_loss / len(progress_bar)
                train_accuracy = self.evaluator.evaluate(self.model, train_data_loader)
                val_accuracy = self.evaluator.evaluate(self.model, test_data_loader)
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Accuracy/test", val_accuracy, epoch)
                self.log_and_save(epoch, val_accuracy, file_name)
        writer.close()
             
    def distillation_train(self, teacher, train_data_loader : DataLoader, test_data_loader : DataLoader, epochs : int, file_name : str, log_dir : str):
        writer = SummaryWriter(log_dir = log_dir)
        teacher.eval().to(self.accelerator.device)
        for param in teacher.parameters():
            param.requires_grad = False
        self.model.train()
        self.model, self.optimizer, train_data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_data_loader, self.scheduler)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                with torch.no_grad():
                    teacher_logits = teacher(x)
                student_logits = self.model(x)
                loss = self.loss_fn(student_logits, teacher_logits, y)

                self.accelerator.backward(loss)
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.scheduler.step()
                epoch_loss = epoch_loss / len(progress_bar)
                train_accuracy = self.evaluator.evaluate(self.model, train_data_loader)
                val_accuracy = self.evaluator.evaluate(self.model, test_data_loader)
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Accuracy/test", val_accuracy, epoch)
                self.log_and_save(epoch, val_accuracy, file_name)
        writer.close()
            
    def log_and_save(self, epoch, val_accuracy, file_name):
        log_string = f"Validation accuracy at epoch {epoch}: {val_accuracy * 100 :.2f}%"
        if self.best_val_accuracy < val_accuracy:
            self.best_val_accuracy = val_accuracy
            log_string += " --> Best model ever (stored)"
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model, file_name + '.pt')
        print(log_string)
