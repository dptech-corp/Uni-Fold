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
from dp.launching.cli import to_runner, default_minimal_exception_handler

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


MIN_SINGLE_SEQUENCE_LENGTH = 16 # to satisfy mmseqs
MAX_SINGLE_SEQUENCE_LENGTH = 3000
MAX_MULTIMER_LENGTH = 3000
PARAM_DIR = "/root/params"


class UnifoldOptions(BaseModel):
    sequence: String = Field(
        min_length=6, max_length=3000,
        description="Input sequence(s). For multimeric targets, please separate different chains with `;`.",
    )
    name: String = Field(
        default="unifold", min_length=0, max_length=31,
        description="Name of the target. "
    )
    symmetry_group: String = Field(default="C1")
    use_template: Boolean = Field(default=True)
    use_msa: Boolean = Field(default=True)
    num_recycling: Int = Field(default=4, ge=1, le=8)
    num_ensembles: Int = Field(default=2, ge=1, le=5)
    num_replica: Int = Field(default=1, ge=1, le=5,
        description="Times of repeatedly running Uni-Fold (with different preprocessing seeds for MSA sampling, etc.)."
    )
    seed: Int = Field(default=0, ge=0)
    output_dir: OutputDirectory = Field(
        default="./output"
    )  # default will be override after online

def main(opts: UnifoldOptions) -> int:
    input_sequences = opts.sequence.strip().split(";")
    symmetry_group = opts.symmetry_group
    output_dir_base = opts.output_dir.get_full_path()
    jobname = re.sub(r'[^a-zA-Z0-9]', '_', opts.name).lower()
    target_id = jobname
    use_templates = opts.use_template
    msa_mode = "MMseqs2" if opts.use_msa else "single_sequence"
    times = opts.num_replica
    num_ensembles = opts.num_ensembles
    num_recycling = opts.num_recycling
    manual_seed = opts.seed

    os.makedirs(output_dir_base, exist_ok=True)

    sequences, is_multimer, symmetry_group = validate_input(
        input_sequences=input_sequences,
        symmetry_group=symmetry_group,
        min_length=MIN_SINGLE_SEQUENCE_LENGTH,
        max_length=MAX_SINGLE_SEQUENCE_LENGTH,
        max_multimer_length=MAX_MULTIMER_LENGTH
    )
    descriptions = ['> '+target_id+' seq'+str(ii) for ii in range(len(sequences))]

    if is_multimer:
        divide_multi_chains(target_id, output_dir_base, sequences, descriptions)

    s = []
    for des, seq in zip(descriptions, sequences):
        s += [des, seq]

    unique_sequences = []
    [unique_sequences.append(x) for x in sequences if x not in unique_sequences]

    if len(unique_sequences)==1:
        homooligomers_num = len(sequences)
    else:
        homooligomers_num = 1
        
    with open(os.path.join(output_dir_base, f"{jobname}.fasta"), "w") as f:
        f.write("\n".join(s))

    result_dir = Path(output_dir_base)
    output_dir = os.path.join(output_dir_base, target_id)

    (
        unpaired_msa,
        paired_msa,
        template_results,
    ) = get_msa_and_templates(
        target_id,
        unique_sequences,
        result_dir=result_dir,
        msa_mode=msa_mode,
        use_templates=use_templates,
        homooligomers_num = homooligomers_num
    )

    for idx, seq in enumerate(unique_sequences):
        chain_id = PDB_CHAIN_IDS[idx]
        sequence_features = pipeline.make_sequence_features(
                sequence=seq, description=f'> {jobname} seq {chain_id}', num_res=len(seq)
            )
        monomer_msa = parsers.parse_a3m(unpaired_msa[idx])
        msa_features = pipeline.make_msa_features([monomer_msa])
        template_features = template_results[idx]
        feature_dict = {**sequence_features, **msa_features, **template_features}
        feature_dict = compress_features(feature_dict)
        features_output_path = os.path.join(
                output_dir, "{}.feature.pkl.gz".format(chain_id)
            )
        pickle.dump(
            feature_dict, 
            gzip.GzipFile(features_output_path, "wb"), 
            protocol=4
            )
        if is_multimer:
            multimer_msa = parsers.parse_a3m(paired_msa[idx])
            pair_features = pipeline.make_msa_features([multimer_msa])
            pair_feature_dict = compress_features(pair_features)
            uniprot_output_path = os.path.join(
                output_dir, "{}.uniprot.pkl.gz".format(chain_id)
            )
            pickle.dump(
                pair_feature_dict,
                gzip.GzipFile(uniprot_output_path, "wb"),
                protocol=4,
            )

    best_result = colab_inference(
        target_id=target_id,
        data_dir=output_dir,
        param_dir=PARAM_DIR,
        output_dir=output_dir,
        symmetry_group=symmetry_group,
        is_multimer=is_multimer,
        max_recycling_iters=num_recycling - 1,
        num_ensembles=num_ensembles,
        times=times,
        manual_seed=manual_seed,
        device="cuda:0",                # do not change this on colab.
    )

    return



def to_parser():
    return to_runner(
        UnifoldOptions,
        main,
        version='0.1.0',
        exception_handler=default_minimal_exception_handler,
    )

if __name__ == '__main__':
    import sys
    to_parser()(sys.argv[1:])
