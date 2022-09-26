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
import shutil
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
flags.DEFINE_string(
    "jackhmmer_binary_path",
    shutil.which("jackhmmer"),
    "Path to the JackHMMER executable.",
)
flags.DEFINE_string(
    "hhblits_binary_path", shutil.which("hhblits"), "Path to the HHblits executable."
)
flags.DEFINE_string(
    "hhsearch_binary_path", shutil.which("hhsearch"), "Path to the HHsearch executable."
)
flags.DEFINE_string(
    "hmmsearch_binary_path",
    shutil.which("hmmsearch"),
    "Path to the hmmsearch executable.",
)
flags.DEFINE_string(
    "hmmbuild_binary_path", shutil.which("hmmbuild"), "Path to the hmmbuild executable."
)
flags.DEFINE_string(
    "kalign_binary_path", shutil.which("kalign"), "Path to the Kalign executable."
)
flags.DEFINE_string(
    "uniref90_database_path",
    None,
    "Path to the Uniref90 database for use by JackHMMER.",
)
flags.DEFINE_string(
    "mgnify_database_path", None, "Path to the MGnify database for use by JackHMMER."
)
flags.DEFINE_string(
    "bfd_database_path", None, "Path to the BFD database for use by HHblits."
)
flags.DEFINE_string(
    "small_bfd_database_path",
    None,
    'Path to the small version of BFD used with the "reduced_dbs" preset.',
)
flags.DEFINE_string(
    "uniclust30_database_path",
    None,
    "Path to the Uniclust30 " "database for use by HHblits.",
)
flags.DEFINE_string(
    "uniprot_database_path",
    None,
    "Path to the Uniprot database for use by JackHMMer.",
)
flags.DEFINE_string(
    "pdb_seqres_database_path",
    None,
    "Path to the PDB seqres database for use by hmmsearch.",
)
flags.DEFINE_string(
    "template_mmcif_dir",
    None,
    "Path to a directory with template mmCIF structures, each named " "<pdb_id>.cif",
)
flags.DEFINE_string(
    "max_template_date",
    None,
    "Maximum template release date to consider. Important if folding "
    "historical test sets.",
)
flags.DEFINE_string(
    "obsolete_pdbs_path",
    None,
    "Path to file containing a mapping from obsolete PDB IDs to the PDB IDs "
    "of their replacements.",
)
flags.DEFINE_enum(
    "db_preset",
    "full_dbs",
    ["full_dbs", "reduced_dbs"],
    "Choose preset MSA database configuration - smaller genetic database "
    "config (reduced_dbs) or full genetic database config  (full_dbs)",
)
flags.DEFINE_boolean(
    "use_precomputed_msas",
    True,
    "Whether to read MSAs that have been written to disk instead of running "
    "the MSA tools. The MSA files are looked up in the output directory, "
    "so it must stay the same between multiple runs that are to reuse the "
    "MSAs. WARNING: This will not check if the sequence, database or "
    "configuration have changed.",
)
flags.DEFINE_boolean("use_uniprot", True, "Whether to use UniProt MSAs.")

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20


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
    data_pipeline: pipeline.DataPipeline,
    use_uniprot: bool,
):
    """
    Predicts structure using AlphaFold for the given sequence.
    """
    logging.info(f"searching homogeneous Sequences & structures for {fasta_name}...")
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name.split("_")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chain_id = fasta_name.split("_")[1] if len(fasta_name.split("_")) > 1 else "A"
    msa_output_dir = os.path.join(output_dir, chain_id)
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    # Get features.
    features_output_path = os.path.join(
        output_dir, "{}.feature.pkl.gz".format(chain_id)
    )
    if not os.path.exists(features_output_path):
        t_0 = time.time()
        feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
        )
        timings["features"] = time.time() - t_0
        feature_dict = compress_features(feature_dict)
        pickle.dump(feature_dict, gzip.GzipFile(features_output_path, "wb"), protocol=4)

    # Get uniprot
    if use_uniprot:
        uniprot_output_path = os.path.join(
            output_dir, "{}.uniprot.pkl.gz".format(chain_id)
        )
        if not os.path.exists(uniprot_output_path):
            t_0 = time.time()
            all_seq_feature_dict = data_pipeline.process_uniprot(
                input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
            )
            timings["all_seq_features"] = time.time() - t_0
            all_seq_feature_dict = compress_features(all_seq_feature_dict)
            pickle.dump(
                all_seq_feature_dict,
                gzip.GzipFile(uniprot_output_path, "wb"),
                protocol=4,
            )

    logging.info("Final timings for %s: %s", fasta_name, timings)

    timings_output_path = os.path.join(output_dir, "{}.timings.json".format(chain_id))
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    for tool_name in (
        "jackhmmer",
        "hhblits",
        "hhsearch",
        "hmmsearch",
        "hmmbuild",
        "kalign",
    ):
        if not FLAGS[f"{tool_name}_binary_path"].value:
            raise ValueError(
                f'Could not find path to the "{tool_name}" binary. Make '
                "sure it is installed on your system."
            )

    use_small_bfd = FLAGS.db_preset == "reduced_dbs"
    _check_flag("small_bfd_database_path", "db_preset", should_be_set=use_small_bfd)
    _check_flag("bfd_database_path", "db_preset", should_be_set=not use_small_bfd)
    _check_flag(
        "uniclust30_database_path", "db_preset", should_be_set=not use_small_bfd
    )

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path,
    )

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
    )

    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniclust30_database_path=FLAGS.uniclust30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
    )

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
            data_pipeline=data_pipeline,
            use_uniprot=FLAGS.use_uniprot,
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_path",
            "output_dir",
            "uniref90_database_path",
            "mgnify_database_path",
            "template_mmcif_dir",
            "max_template_date",
            "obsolete_pdbs_path",
        ]
    )

    app.run(main)
