import numpy as np
import torch

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLossNegCon(nn.Module):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.probs_temp = 0.1
        self.center_momentum = center_momentum
        self.probs_momentum = 0.998
        self.batchsize = batchsize
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("probs_pos", torch.ones(1, out_dim) / out_dim)
        self.register_buffer("probs_neg", torch.ones(1, out_dim) / out_dim)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, crops_freq_student, crops_freq_teacher, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # total number of crops
        n_crops_student = len(student_output) // self.batchsize
        n_crops_teacher = len(teacher_output) // self.batchsize

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(n_crops_student)
        temp = self.teacher_temp_schedule[epoch]
        teacher_probs = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_probs.detach().chunk(n_crops_teacher)
        out_dim = teacher_out[0].shape[-1]
        total_loss = 0.
        n_loss_terms = 0
        end_idx_s = torch.cumsum(torch.tensor(crops_freq_student), 0)
        for t in range(crops_freq_teacher[0]):  # loop over pos teacher crops
            start_s = 0
            for k, end_s in enumerate(end_idx_s):  # loop over datasets
                for s in range(start_s, end_s):  # loop over student crops
                    # pos loss
                    if k == 0:
                        if s == t:  # we skip cases where student and teacher operate on the same view
                            continue
                        loss = torch.sum(-teacher_out[t] * F.log_softmax(student_out[s], dim=-1), dim=-1)
                        # in-dist neg loss
                    elif k == 1:
                        loss = 0.5 / out_dim * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)
                        # loss = 0.5/out_dim * torch.sum(-(1.-teacher_out[t]) * F.log_softmax(student_out[s], dim=-1), dim=-1)
                        # loss = 0.5/out_dim * (1.-probs_pos) * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)
                    # aux neg loss
                    else:
                        loss = 0.5 / out_dim * torch.sum(-F.log_softmax(student_out[s], dim=-1), dim=-1)
                        # loss = 0.5/out_dim * torch.sum(-(1.-teacher_out[t]) * F.log_softmax(student_out[s], dim=-1), dim=-1)
                    total_loss += len(crops_freq_student) * loss.mean()  # scaling loss with batchsize
                    n_loss_terms += 1
                start_s = end_s
        total_loss /= n_loss_terms
        self.center = self.update_ema(teacher_output[:crops_freq_teacher[0]], self.center, self.center_momentum)
        self.probs_pos = self.update_ema(teacher_probs[:crops_freq_teacher[0]], self.probs_pos, self.probs_momentum)
        self.probs_neg = self.update_ema(teacher_probs[crops_freq_teacher[0]:], self.probs_neg, self.probs_momentum)
        return total_loss

    @torch.no_grad()
    def update_ema(self, output, ema, momentum):
        """
        Update exponential moving aveages for teacher output.
        """
        batch_center = torch.sum(output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(output) * dist.get_world_size())

        # ema update
        return ema * momentum + batch_center * (1 - momentum)


class DINOLoss_vanilla(nn.Module):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.class_probs_momentum = 0.998
        self.batchsize = batchsize
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        num_crops_student = student_output.shape[0] // self.batchsize
        num_crops_teacher = teacher_output.shape[0] // self.batchsize
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(num_crops_student)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(num_crops_teacher)
        
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)




class DINOLoss_classprob_v2(nn.Module):
    def __init__(self, out_dim, batchsize, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.class_probs_momentum = 0.998
        self.batchsize = batchsize
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("class_probs", torch.ones(1, out_dim) / out_dim)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        num_crops_student = student_output.shape[0] // self.batchsize
        num_crops_teacher = teacher_output.shape[0] // self.batchsize
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(num_crops_student)

        temp = self.teacher_temp_schedule[epoch]
        teacher_probs = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_probs.detach().chunk(num_crops_teacher)
        total_loss = 0
        n_loss_terms = 0
        num_class = teacher_out[0].shape[-1]
        if temp <= self.student_temp:
            for iq, qt in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view for positive examples
                        continue
                    loss = torch.sum(-qt * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
        else:
            for v in range(len(student_out)):
                loss = 1 / num_class * torch.sum(-(1. - self.class_probs) * F.log_softmax(student_out[v], dim=-1),
                                                 dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        self.update_center(teacher_output)
        self.update_class_probs(teacher_probs)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def update_class_probs(self, teacher_probs):
        """
        Update teacher probabilities.
        """
        batch_class_probs = torch.sum(teacher_probs, dim=0, keepdim=True)
        dist.all_reduce(batch_class_probs)
        batch_class_probs = batch_class_probs / (len(teacher_probs) * dist.get_world_size())

        # ema update
        self.class_probs = self.class_probs * self.class_probs_momentum + batch_class_probs * (
                    1 - self.class_probs_momentum)
