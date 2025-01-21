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
                 best_loss = float("inf"),
                 log_dir: str = "runs"
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.accelerator = Accelerator(mixed_precision = 'no')
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.writer = SummaryWriter(log_dir=log_dir)
        self.evaluator = Evaluator(self.accelerator.device)

    def train(self, train_data_loader : DataLoader, test_data_loader : DataLoader, epochs : int, file_name : str):
        self.model.train()
        self.model, self.optimizer, train_data_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_data_loader)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(train_data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                self.optimizer.zero_grad()
                x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)

                self.accelerator.backward(loss)
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            epoch_loss = epoch_loss / len(progress_bar)
            self.writer.add_scalar("Loss/train", epoch_loss.item(), epoch)
            self.writer.add_scalar("Accuracy/test", self.evaluator.evaluate(self.model, test_data_loader), epoch)
            self.log_and_save(epoch, epoch_loss, file_name)
        self.writer.close()
             
    def distillation_train(self, teacher, data_loader : DataLoader, test_data_loader : DataLoader, epochs : int, file_name : str):
        teacher.eval().to(self.accelerator.device)
        for param in teacher.parameters():
            param.requires_grad = False
        self.model.train()
        self.model, self.optimizer, data_loader = self.accelerator.prepare(
            self.model, self.optimizer, data_loader)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
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
                
            epoch_loss = epoch_loss / len(progress_bar)
            self.writer.add_scalar("Loss/train", epoch_loss.item(), epoch)
            self.writer.add_scalar("Accuracy/test", self.evaluator.evaluate(self.model, test_data_loader), epoch)
            self.log_and_save(epoch, epoch_loss, file_name)
        self.writer.close()
            
    def log_and_save(self, epoch, epoch_loss, file_name):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
                
            log_string = f"Loss at epoch {epoch}: {epoch_loss :.4f}"
            if self.best_loss > epoch_loss:
                self.best_loss = epoch_loss
                torch.save({
                    "model_state_dict": self.accelerator.get_state_dict(self.model),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    }, file_name + '_epoch' + str(epoch) + '.pth')
                log_string += " --> Best model ever (stored)"
            print(log_string)