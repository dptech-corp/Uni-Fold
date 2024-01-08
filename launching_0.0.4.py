# Copyright 2022 DP Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
entrypoint for Uni-Fold @ Bohrium Apps. 
temporary URL: https://labs.dp.tech/projects/unifold/
"""

import sys

from dp.launching.typing import (
    BaseModel,
    Field,
    Int,
    String,
    Boolean,
    OutputDirectory,
)
from dp.launching.cli import (
    to_runner,
    default_minimal_exception_handler,
    default_exception_handler
)

import os
import re
import random
import hashlib
import numpy as np
from pathlib import Path
from unifold.colab.data import validate_input
from unifold.msa.utils import divide_multi_chains

import pickle
import gzip
from unifold.msa import parsers
from unifold.msa import pipeline
from unifold.data.utils import compress_features
from unifold.data.protein import PDB_CHAIN_IDS
from unifold.colab.mmseqs import get_msa_and_templates
from unifold.colab.model import colab_inference

from unifold.launching.data import (
    parse_batch_inputs,
    valid_symmetry_group,
    get_msa_and_templates,
)
from unifold.launching.inf import launching_inference


MIN_SINGLE_SEQUENCE_LENGTH = 16 # to satisfy mmseqs
MAX_SINGLE_SEQUENCE_LENGTH = 3000
MAX_MULTIMER_LENGTH = 3000
PARAM_DIR = "/root/params"


class UnifoldOptions(BaseModel):
    sequence: String = Field(
        default="MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        description="Input sequence(s). Each line should contain one prediction target. For multimeric targets, please separate different chains with `;`.",
    )
    symmetry_group: String = Field(default="C1")
    num_replica: Int = Field(default=1, ge=1, le=5,
        description="Times of repeatedly running Uni-Fold (with different preprocessing seeds for MSA sampling, etc.)."
    )
    num_recycling: Int = Field(default=4, ge=1, le=8,
        description="Times of resampling MSA clusters in the AlphaFold pipeline. Generally does not have significant impact on the output structure."
    )
    num_ensembles: Int = Field(default=2, ge=1, le=5,
        description="Times of resampling MSA clusters in the AlphaFold pipeline. Generally does not have significant impact on the output structure."
    )
    seed: Int = Field(default=0, ge=0,
        description="Random seed for the entire task. Notably new seeds will be generated for each replica."
    )
    use_msa: Boolean = Field(default=True,
        description="Whether to use MSAs to predict the structure. Setting this as TRUE is highly recommended."
    )
    use_template: Boolean = Field(default=True,
        description="Whether to use structural templates (homologous structures) in the modelling process. Only useful when `use_msa` is set as TRUE."
    )
    use_multimer: Boolean = Field(default=True,
        description="Whether to use `multimer_ft` as the base model. Must be TRUE if the target contains multimeric case. Ignored if symmetry group is not C1."
    )
    output_dir: OutputDirectory = Field(
        default="./output"
    )  # default will be override after online


def main(opts: UnifoldOptions) -> int:
    if ";" in opts.sequence and not opts.use_multimer:
        raise ValueError("must set `use_multimer` as TRUE if multimeric cases are inputted.")
    output_dir = opts.output_dir.get_full_path()
    # parse queries
    all_targets, seqid_map = parse_batch_inputs(
        opts.sequence,
        min_length=0,
        max_length=3000,
    )
    print(f"parsed queries: {all_targets}")
    print(f"unique query sequences: " + str({k: v[:5]+"..." for k, v in seqid_map.items()}))
    # clean symmetry group
    symmetry_group = valid_symmetry_group(
        opts.symmetry_group,
    )
    print(f"symmetry group: {symmetry_group} (`C1` -> None)")
    # mmseqs
    feat_dir = get_msa_and_templates(
        seqid_map,
        output_dir,
        use_msa=opts.use_msa,
        use_templates=opts.use_template,
        mmseqs_api=None,
        chunk_size=100,
    )
    print("mmseqs features prepared.")
    # model inference
    result_out_dir = os.path.join(output_dir, "prediction")
    launching_inference(
        all_targets,
        feat_dir=feat_dir,
        param_dir=PARAM_DIR,
        prediction_dir=result_out_dir,
        use_multimer=opts.use_multimer,
        symmetry_group=symmetry_group,
        max_recycling_iters=opts.num_recycling,
        num_ensembles=opts.num_ensembles,
        times=opts.num_replica,
        manual_seed=opts.seed,
        device="cuda",
    )
    print("all targets done.")
    return


def to_parser():
    return to_runner(
        UnifoldOptions,
        main,
        version='0.1.0',
        exception_handler=default_exception_handler,
    )

if __name__ == '__main__':
    import sys
    to_parser()(sys.argv[1:])
