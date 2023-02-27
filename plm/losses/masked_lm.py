
import logging
import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss

logger = logging.getLogger(__name__)

@register_loss("esm_masked_lm")
class ESMMaskedLMLoss(UnicoreLoss):
    def __init__(self, task, mlm_alpha, freq_alpha, reduce22, l1loss):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.mlm_alpha = mlm_alpha
        self.freq_alpha = freq_alpha
        self.reduce22 = reduce22
        self.l1loss = l1loss

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""        
        parser.add_argument(
            "--mlm-alpha",
            default=1.0,
            type=float,
            help="default: 1.0 \
            loss = mlm_alpha * mlm_loss + freq_alpha * freq_loss"
        )
        parser.add_argument(
            "--freq-alpha",
            default=1.0,
            type=float,
            help="default: 1.0 \
            loss = mlm_alpha * mlm_loss + freq_alpha * freq_loss"
        )
        parser.add_argument(
            "--reduce22",
            default=False,
            action='store_true',
            help="whether to reduce loss in the freq dim"
        )
        parser.add_argument(
            "--l1loss",
            default=False,
            action='store_true',
            help="whether to use l1 distance as loss function"
        )

    def forward(self, model, sample, reduce=True):
        masked_tokens = sample["target"].ne(self.padding_idx) # B x L

        sample_size = masked_tokens.int().sum()

        masked_tokens = torch.where(
            masked_tokens.any(),
            masked_tokens,
            masked_tokens.new([True]),
        )
        logits = model(**sample["net_input"], masked_tokens=masked_tokens)
        target = sample['target']
        
        if masked_tokens is not None:
            target = target[masked_tokens]
            # freq_target = freq_target[masked_tokens]
        
        mlm_preds = torch.argmax(logits, dim=-1)
        mlm_acc = 1. * torch.sum(mlm_preds==target)    
        
        mlm_loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction='sum',
        )
        
        total_loss = self.mlm_alpha * mlm_loss / sample_size 
        
        
        logging_output = {
            "loss": total_loss.data,
            "mlm_loss": self.mlm_alpha*mlm_loss.data / sample_size,
            "bsz": sample["target"].size(0),
            "sample_size": 1,
            "seq_len": sample["target"].size(1) * sample["target"].size(0),
            "mlm_acc": mlm_acc / sample_size,
        }
        return total_loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        mlm_acc_sum = sum(log.get("mlm_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
      
        metrics.log_scalar("mlm_acc", mlm_acc_sum / sample_size, sample_size, round=3,)
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
