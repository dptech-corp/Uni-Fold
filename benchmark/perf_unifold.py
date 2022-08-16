import argparse
import os

import torch
import torch.nn as nn

import os
import sys
import pathlib

from tqdm import tqdm

from unifold.modules.evoformer import EvoformerIteration
from unifold.modules.attentions import gen_msa_attn_mask, gen_tri_attn_mask


class WrapEvoformerLayer(nn.Module):
    def __init__(self, d_node, d_pair):
        super(WrapEvoformerLayer, self).__init__()
        self.d_node = d_node
        self.d_pair = d_pair

        self.c_hidden_msa_att = int(d_node / 8)
        self.c_hidden_pair_att = int(d_pair / 4)

        self.EvoformerIteration = EvoformerIteration(
            d_msa=d_node,
            d_pair=d_pair,
            d_hid_msa_att=self.c_hidden_msa_att,
            d_hid_opm=self.c_hidden_msa_att,
            d_hid_mul=self.d_pair,
            d_hid_pair_att=self.c_hidden_pair_att,
            num_heads_msa=8,
            num_heads_pair=4,
            transition_n=4,
            msa_dropout=0.15,
            pair_dropout=0.25,
            outer_product_mean_first=False,
            inf=3e4,
            eps=1e-5,
        )
        self.alphafold_original_mode()

    def alphafold_original_mode(self):
        def set_alphafold_original_mode(module):
            if hasattr(module, "apply_alphafold_original_mode"):
                module.apply_alphafold_original_mode()
            if hasattr(module, "act"):
                module.act = nn.ReLU()

        self.apply(set_alphafold_original_mode)

    def forward(
        self,
        node,
        pair,
        node_mask,
        pair_mask,
        msa_row_attn_mask: torch.Tensor,
        msa_col_attn_mask: torch.Tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: int,
    ):
        node, pair = self.EvoformerIteration(
            node,
            pair,
            node_mask,
            pair_mask,
            msa_row_attn_mask,
            msa_col_attn_mask,
            tri_start_attn_mask,
            tri_end_attn_mask,
            chunk_size=chunk_size,
        )
        return node, pair


def main():

    parser = argparse.ArgumentParser(description="Evoformer Standalone Perf Benchmark")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--msa-length", default=128, type=int, help="Sequence Length of MSA"
    )
    parser.add_argument(
        "--res-length", default=256, type=int, help="Sequence Length of Residues"
    )
    parser.add_argument(
        "--trials", default=50, type=int, help="Number of Trials to Execute"
    )
    parser.add_argument(
        "--warmup-trials", default=5, type=int, help="Warmup Trials to discard"
    )
    parser.add_argument(
        "--layers", default=4, type=int, help="Evoformer Layers to Execute"
    )
    parser.add_argument(
        "--chunk-size", default=None, type=int, help="Evoformer Layers to Execute"
    )
    parser.add_argument("--cm", default=256, type=int, help="MSA hidden dimension")
    parser.add_argument("--cz", default=128, type=int, help="Pair hidden dimension")
    parser.add_argument("--fwd", action="store_true", help="Only execute Fwd Pass.")
    parser.add_argument(
        "--fp16", action="store_true", help="Use fp16 for benchmark (for V100)"
    )

    args = parser.parse_args()

    precision = torch.bfloat16

    if args.fp16:
        precision = torch.float16

    if not torch.cuda.is_available():
        raise NotImplementedError("Running on CPU is not supported")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    attn_layers = []
    for idx in range(0, args.layers):
        attn_layers.append(WrapEvoformerLayer(d_node=args.cm, d_pair=args.cz))
        attn_layers[idx].cuda()
        attn_layers[idx].to(dtype=precision)
        if args.fwd:
            attn_layers[idx].eval()

    start_evt_fwd = []
    start_evt_bwd = []
    stop_evt_bwd = []
    for recorded_trial in range(0, args.trials):
        start_evt_fwd.append(torch.cuda.Event(enable_timing=True))
        start_evt_bwd.append(torch.cuda.Event(enable_timing=True))
        stop_evt_bwd.append(torch.cuda.Event(enable_timing=True))

    inputs_node = torch.randn(
        args.batch_size,
        args.msa_length,
        args.res_length,
        args.cm,
        dtype=precision,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    inputs_pair = torch.randn(
        args.batch_size,
        args.res_length,
        args.res_length,
        args.cz,
        dtype=precision,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    node_mask = torch.ones(
        (args.batch_size, args.msa_length, args.res_length),
        dtype=precision,
        device=torch.device("cuda"),
    ).requires_grad_(False)

    msa_raw_mask, msa_col_mask = gen_msa_attn_mask(node_mask, 3e4)
    pair_mask = torch.ones(
        (args.batch_size, args.res_length, args.res_length),
        dtype=precision,
        device=torch.device("cuda"),
    ).requires_grad_(False)
    tri_start_mask, tri_end_mask = gen_tri_attn_mask(pair_mask, 3e4)

    total_used_mem_gb = 0
    for trial in range(0, args.trials + args.warmup_trials):
        layer_inputs = inputs_node, inputs_pair
        evt_idx = trial - args.warmup_trials

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if evt_idx >= 0:
            start_evt_fwd[evt_idx].record()
        with torch.set_grad_enabled(not args.fwd):
            for lyr_idx in range(0, args.layers):
                layer_inputs = attn_layers[lyr_idx].forward(
                    *layer_inputs,
                    node_mask,
                    pair_mask,
                    msa_raw_mask,
                    msa_col_mask,
                    tri_start_mask,
                    tri_end_mask,
                    chunk_size=args.chunk_size,
                )

        torch.cuda.synchronize()

        if evt_idx >= 0:
            start_evt_bwd[evt_idx].record()

        if not args.fwd:
            s = layer_inputs[0].mean() + layer_inputs[1].mean()
            s.backward()

        torch.cuda.synchronize()
        cur_cost_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        total_used_mem_gb += cur_cost_mem
        if evt_idx >= 0:
            stop_evt_bwd[evt_idx].record()


    torch.cuda.synchronize()
    elapsed_time_fwd = 0.0
    elapsed_time_bwd = 0.0
    for evt_idx in range(0, args.trials):
        elapsed_time_fwd += start_evt_fwd[evt_idx].elapsed_time(start_evt_bwd[evt_idx])
        elapsed_time_bwd += start_evt_bwd[evt_idx].elapsed_time(stop_evt_bwd[evt_idx])

    print(
        " Input: {:4d}, {:4d}, {:4d}, ({:4d} {:4d}), Fwd Time / Layer: {:.3f} ms, Bwd Time / Layer: {:.3f} ms, Memory cost {:.3f} GB".format(
            args.batch_size,
            args.msa_length,
            args.res_length,
            args.cm,
            args.cz,
            elapsed_time_fwd  / (args.trials * args.layers),
            elapsed_time_bwd  / (args.trials * args.layers),
            total_used_mem_gb / (args.trials + args.warmup_trials),
        )
    )


if __name__ == "__main__":
    main()
