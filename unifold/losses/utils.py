import torch
from unifold.data import residue_constants as rc


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits.float(), dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits = logits.float()
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-logits)
    loss = -labels * log_p - (1 - labels) * log_not_p
    return loss


def masked_mean(mask, value, dim, eps=1e-10, keepdim=False):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
        eps + torch.sum(mask, dim=dim, keepdim=keepdim)
    )
