import logging
import torch

from unicore import metrics
from unicore.utils import tensor_tree_map
from unicore.losses import UnicoreLoss, register_loss
from unicore.data import data_utils

from unifold.losses.geometry import compute_renamed_ground_truth, compute_metric
from unifold.losses.violation import find_structural_violations, violation_loss
from unifold.losses.fape import fape_loss
from unifold.losses.auxillary import (
    chain_centre_mass_loss,
    distogram_loss,
    experimentally_resolved_loss,
    masked_msa_loss,
    pae_loss,
    plddt_loss,
    repr_norm_loss,
    masked_msa_loss,
    supervised_chi_loss,
)
from unifold.losses.chain_align import multi_chain_perm_align


@register_loss("af2")
class AlphafoldLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, batch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # return config in model.
        out, config = model(batch)
        num_recycling = batch["msa_feat"].shape[0]
        
        # remove recyling dim
        batch = tensor_tree_map(lambda t: t[-1, ...], batch)
        
        loss, sample_size, logging_output = self.loss(out, batch, config)
        logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output

    def loss(self, out, batch, config):

        if "violation" not in out.keys() and config.violation.weight:
            out["violation"] = find_structural_violations(
                batch, out["sm"]["positions"], **config.violation)

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(batch, out["sm"]["positions"]))
        
        loss_dict = {}
        loss_fns = {
            "chain_centre_mass": lambda: chain_centre_mass_loss(
                pred_atom_positions=out["final_atom_positions"],
                true_atom_positions=batch["all_atom_positions"],
                atom_mask=batch["all_atom_mask"],
                asym_id=batch["asym_id"],
                **config.chain_centre_mass,
                loss_dict=loss_dict,
            ),
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                pseudo_beta=batch["pseudo_beta"],
                pseudo_beta_mask=batch["pseudo_beta_mask"],
                **config.distogram,
                loss_dict=loss_dict,
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                atom37_atom_exists=batch["atom37_atom_exists"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.experimentally_resolved,
                loss_dict=loss_dict,
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                config.fape,
                loss_dict=loss_dict,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                true_msa=batch["true_msa"],
                bert_mask=batch["bert_mask"],
                loss_dict=loss_dict,
            ),
            "pae": lambda: pae_loss(
                logits=out["pae_logits"],
                pred_frame_tensor=out["pred_frame_tensor"],
                true_frame_tensor=batch["true_frame_tensor"],
                frame_mask=batch["frame_mask"],
                resolution=batch["resolution"],
                **config.pae,
                loss_dict=loss_dict,
            ),
            "plddt": lambda: plddt_loss(
                logits=out["plddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.plddt,
                loss_dict=loss_dict,
            ),
            "repr_norm": lambda: repr_norm_loss(
                out["delta_msa"],
                out["delta_pair"],
                out["msa_norm_mask"],
                batch["pseudo_beta_mask"],
                **config.repr_norm,
                loss_dict=loss_dict,
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                pred_angles_sin_cos=out["sm"]["angles"],
                pred_unnormed_angles_sin_cos=out["sm"]["unnormalized_angles"],
                true_angles_sin_cos=batch["chi_angles_sin_cos"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                chi_mask=batch["chi_mask"],
                **config.supervised_chi,
                loss_dict=loss_dict,
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                loss_dict=loss_dict,
                bond_angle_loss_weight=config.violation.bond_angle_loss_weight,
            ),
        }

        cum_loss = 0
        bsz = batch["seq_mask"].shape[0]
        with torch.no_grad():
            seq_len = torch.sum(batch["seq_mask"].float(), dim=-1)
            seq_length_weight = seq_len**0.5
        
        assert (
            len(seq_length_weight.shape) == 1 and seq_length_weight.shape[0] == bsz
        ), seq_length_weight.shape
        
        for loss_name, loss_fn in loss_fns.items():
            weight = config[loss_name].weight
            if weight > 0.:
                loss = loss_fn()
                # always use float type for loss
                assert loss.dtype == torch.float, loss.dtype
                assert len(loss.shape) == 1 and loss.shape[0] == bsz, loss.shape

                if any(torch.isnan(loss)) or any(torch.isinf(loss)):
                    logging.warning(f"{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0.0, requires_grad=True)
                
                cum_loss = cum_loss + weight * loss

        for key in loss_dict:
            loss_dict[key] = float((loss_dict[key]).mean())

        loss = (cum_loss * seq_length_weight).mean()

        logging_output = loss_dict
        # sample size fix to 1, so the loss (and gradients) will be averaged on all workers.
        sample_size = 1
        logging_output["loss"] = loss.data
        logging_output["bsz"] = bsz
        logging_output["sample_size"] = sample_size
        logging_output["seq_len"] = seq_len
        # logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        for key in logging_outputs[0]:
            if key in ["sample_size", "bsz"]:
                continue
            loss_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, loss_sum / sample_size, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_loss("afm")
class AlphafoldMultimerLoss(AlphafoldLoss):
    def forward(self, model, batch, reduce=True):
        features, labels = batch
        assert isinstance(features, dict)

        # return config in model.
        out, config = model(features)
        num_recycling = features["msa_feat"].shape[0]
        
        # remove recycling dim
        features = tensor_tree_map(lambda t: t[-1, ...], features)
        
        # perform multi-chain permutation alignment.
        if labels:
            with torch.no_grad():
                batch_size = out["final_atom_positions"].shape[0]
                new_labels = []
                for batch_idx in range(batch_size):
                    cur_out = {
                        k: out[k][batch_idx]
                        for k in out
                        if k in {"final_atom_positions", "final_atom_mask"}
                    }
                    cur_feature = {k: features[k][batch_idx] for k in features}
                    cur_label = labels[batch_idx]
                    cur_new_labels = multi_chain_perm_align(
                        cur_out, cur_feature, cur_label
                    )
                    new_labels.append(cur_new_labels)
            new_labels = data_utils.collate_dict(new_labels, dim=0)
            
            # check for consistency of label and feature.
            assert (new_labels["aatype"] == features["aatype"]).all()
            features.update(new_labels)

        loss, sample_size, logging_output = self.loss(out, features, config)
        logging_output["num_recycling"] = num_recycling
        
        return loss, sample_size, logging_output
