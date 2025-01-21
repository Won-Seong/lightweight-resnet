import torch
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=1)

        # KL Divergence Loss (Soft Target)
        distillation_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)

        # CrossEntropy Loss (Ground Truth)
        ce_loss = self.ce_loss(student_logits, labels)

        return self.alpha * distillation_loss + (1 - self.alpha) * ce_loss