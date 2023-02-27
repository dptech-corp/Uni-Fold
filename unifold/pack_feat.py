# Copyright 2022 DP Technology
# Copyright 2021 DeepMind Technologies Limited
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

"""Run CPU MSA & template searching to get pickled features."""
import json
import os
import pickle
from pathlib import Path
import time
import gzip

from absl import app
from absl import flags
from absl import logging

from unifold.data.utils import compress_features
from unifold.msa import parsers
from unifold.msa import pipeline
from unifold.msa import templates
from unifold.msa.utils import divide_multi_chains
from unifold.msa.tools import hmmsearch
from unifold.msa.pipeline import make_sequence_features

logging.set_verbosity(logging.INFO)

flags.DEFINE_string(
    "fasta_path",
    None,
    "Path to FASTA file, If a FASTA file contains multiple sequences, "
    "then it will be divided into several single sequences. ",
)

flags.DEFINE_string(
    "output_dir", None, "Path to a directory that will " "store the results."
)

FLAGS = flags.FLAGS



def _check_flag(flag_name: str, other_flag_name: str, should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = "be" if should_be_set else "not be"
        raise ValueError(
            f"{flag_name} must {verb} set when running with "
            f'"--{other_flag_name}={FLAGS[other_flag_name].value}".'
        )


def generate_pkl_features(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
):
    """
    Predicts structure using AlphaFold for the given sequence.
    """
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name.split("_")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chain_id = fasta_name.split("_")[1] if len(fasta_name.split("_")) > 1 else "A"
    input_seqs, input_descs = parsers.parse_fasta(open(fasta_path, "r").read())
    assert len(input_seqs) == 1 
    input_seq = input_seqs[0]
    input_desc = input_descs[0]
    num_res = len(input_seq)
    feature_dict = make_sequence_features(input_seq, input_desc, num_res)
    

    # Get features.
    features_output_path = os.path.join(
        output_dir, "{}.feature.pkl.gz".format(fasta_name)
    )
    if not os.path.exists(features_output_path):
        t_0 = time.time()
        
        timings["features"] = time.time() - t_0
        feature_dict = compress_features(feature_dict)
        pickle.dump(feature_dict, gzip.GzipFile(features_output_path, "wb"), protocol=4)

    

    logging.info("Final timings for %s: %s", fasta_name, timings)

    timings_output_path = os.path.join(output_dir, "{}.timings.json".format(fasta_name))
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    fasta_path = FLAGS.fasta_path
    fasta_name = Path(fasta_path).stem
    input_fasta_str = open(fasta_path).read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) > 1:
        temp_names, temp_paths = divide_multi_chains(
            fasta_name, FLAGS.output_dir, input_seqs, input_descs
        )
        fasta_names = temp_names
        fasta_paths = temp_paths
    else:
        output_dir = os.path.join(FLAGS.output_dir, fasta_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        chain_order_path = os.path.join(output_dir, "chains.txt")
        with open(chain_order_path, "w") as f:
            f.write("A")
        fasta_names = [fasta_name]
        fasta_paths = [fasta_path]

    # Check for duplicate FASTA file names.
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError("All FASTA paths must have a unique basename.")

    # Predict structure for each of the sequences.
    for i, fasta_path in enumerate(fasta_paths):
        fasta_name = fasta_names[i]
        generate_pkl_features(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=FLAGS.output_dir,
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_path",
            "output_dir",
        ]
    )

    app.run(main)
