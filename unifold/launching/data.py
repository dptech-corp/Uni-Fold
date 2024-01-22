from unifold.colab.data import clean_and_validate_sequence
from unifold.colab.mmseqs import get_null_template
from unifold.msa import pipeline, parsers, templates
from unifold.msa.tools import hhsearch
from unifold.data.utils import compress_features
from typing import *
from .mmseqs_api import MMSeqs2API, DEFAULT_API_SERVER
import re
import os, glob
import os.path as osp
from tqdm import tqdm
import pickle, gzip

def unique_seq_id_gen():
    i = 0
    while True:
        i += 1
        yield f"UFS{i:04d}"

def unique_job_id_gen():
    i = 0
    while True:
        i += 1
        yield f"J{i:04d}"

def valid_symmetry_group(symmetry_group:str):
    if not symmetry_group:
        return None
    symmetry_group = symmetry_group.strip().upper()
    if symmetry_group == "C1":  # in C1 case we do not use symmetry model, though we can.
        return None
    if not re.match(r"^C\d+$", symmetry_group):
        raise ValueError(f"Uni-Fold only supports cyclic symmetry groups (`C`). Got group {symmetry_group} instead.")
    return symmetry_group

def parse_batch_inputs(inputs:str, min_length=16, max_length=3000) -> Tuple[List[List[str]], Dict[str, str]]:
    sid_gen = unique_seq_id_gen()
    lines = inputs.strip().split("\n")

    unique_seqs = {}
    all_targets = []
    for line in lines:
        if not len(line):       # ignore empty line
            continue
        sequences = line.strip().split(";")
        seqids = []
        for seq in sequences:
            clean_seq = clean_and_validate_sequence(
                seq, min_length=min_length, max_length=max_length
            )
            if clean_seq not in unique_seqs:
                sid = next(sid_gen)
                unique_seqs[clean_seq] = sid
            else:
                sid = unique_seqs[clean_seq]
            seqids.append(sid)
        all_targets.append(seqids)
    seqid_map = {v: k for k, v in unique_seqs.items()}

    return all_targets, seqid_map

def make_chunked_fastas(seqid_map, chunk_size=None):
    seqs = list(seqid_map.items())
    csz = chunk_size or len(seqid_map)
    ret = []
    for i in range(0, len(seqid_map), csz):
        chunk = seqs[i:i+csz]
        fasta_text = "\n".join([
            f">{sid}\n{seq}" for sid, seq in chunk
        ])
        ret.append(fasta_text)
    return ret

def parse_mmseqs_a3ms(result_dir, output_dir):
    a3m_paths = glob.glob(osp.join(result_dir, "*", "*.a3m"))
    for a3m in tqdm(a3m_paths, total=len(a3m_paths)):
        update_sid, out_a3m_path = True, None
        for line in open(a3m,"r"):
            if not len(line):
                continue
            if "\x00" in line:
                line = line.replace("\x00","")
                update_sid = True
            if line.startswith(">") and update_sid:
                sid = line[1:].rstrip()
                out_a3m_path = osp.join(output_dir, f"{sid}.a3m")
                update_sid = False
            with open(out_a3m_path, "a") as f:
                f.write(line)

def templates_exist_at_path(template_path):
    if not osp.exists(template_path):   # no folder
        return False
    cifs = glob.glob(osp.join(template_path, "*.cif"))
    if not len(cifs):   # no cif
        print(f"no cif found in {template_path}. use null template.", flush=True)
        return False
    return True

def get_template_features(
    a3m_path: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    if not templates_exist_at_path(template_path):   # no tmpl detected
        return get_null_template(query_sequence)
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query_with_a3m_file(a3m_path)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)

def get_msa_and_templates(
    seqid_map: Dict[str, str],
    result_dir: str,
    use_msa: bool = True,
    use_templates: bool = True,
    mmseqs_api: MMSeqs2API = None,
    chunk_size: int = None,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    if use_templates and not use_msa:
        raise ValueError("must choose 'use_msa' if 'use_templates' is set.")
    if mmseqs_api is None:
        mmseqs_api = MMSeqs2API(
            host_url=DEFAULT_API_SERVER,
        )

    # make dirs
    raw_dir = osp.join(result_dir, "raw")
    msa_dir = osp.join(result_dir, "msa")
    tmpl_dir = osp.join(result_dir, "tmpl")
    feat_dir = osp.join(result_dir, "feat")
    for d in (raw_dir, msa_dir, tmpl_dir, feat_dir):
        os.makedirs(d, exist_ok=True)

    # filter for seqs that need msa
    if use_msa:
        mmseq_seqid_map = {}
        nomsa_seqid_map = {}
        for k, v in seqid_map.items():
            if len(v) > 16:
                mmseq_seqid_map[k] = v
            else:
                nomsa_seqid_map[k] = v
    else:
        mmseq_seqid_map = {}
        nomsa_seqid_map = seqid_map

    # make text output files
    if len(mmseq_seqid_map):
        chunked_fastas = make_chunked_fastas(mmseq_seqid_map, chunk_size)
        for ci, fasta in enumerate(chunked_fastas):
            # dump query fasta
            open(osp.join(raw_dir, f"query_{ci:02d}.fasta"), "w").write(fasta)
            # download mmseqs api results
            tgz_path = osp.join(raw_dir, f"out_{ci:02d}.tar.gz")
            if not osp.exists(tgz_path):    # skip this if file already exist.
                mmseqs_api.query(fasta, tgz_path)
            # extract returned tgz
            extract_subdir = osp.join(raw_dir, f"out_{ci:02d}")
            os.makedirs(extract_subdir, exist_ok=True)
            os.system(f"tar zxvf {tgz_path} -C {extract_subdir}")
            # extract templates
            if use_templates:
                mmseqs_api.get_templates(
                    osp.join(extract_subdir, "pdb70.m8"), tmpl_dir,
                )
        # parse all a3m files
        parse_mmseqs_a3ms(raw_dir, msa_dir)

    # make pseudo a3m files for nomsa targets
    for sid, seq in nomsa_seqid_map.items():
        with open(osp.join(msa_dir, f"{sid}.a3m"), "w") as f:
            f.write(f">{sid}\n{seq}\n")

    # make feature files
    for sid, seq in tqdm(seqid_map.items(), total=len(seqid_map)):
        a3m_path = osp.join(msa_dir, f"{sid}.a3m")
        sequence_features = pipeline.make_sequence_features(
            sequence=seq, description=sid, num_res=len(seq)
        )
        monomer_msa = parsers.parse_a3m(open(a3m_path).read())
        msa_features = pipeline.make_msa_features([monomer_msa])
        template_features = get_template_features(
            a3m_path, osp.join(tmpl_dir, sid), seq
        )

        feature_dict = {**sequence_features, **msa_features, **template_features}
        feature_dict = compress_features(feature_dict)
        features_output_path = osp.join(
            feat_dir, f"{sid}.feature.pkl.gz"
        )
        pickle.dump(
            feature_dict, 
            gzip.GzipFile(features_output_path, "wb"), 
            protocol=4
        )

    return feat_dir

