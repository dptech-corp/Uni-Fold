import os
import json
import ml_collections as mlc
import numpy as np
import copy
import torch
from datetime import datetime
from typing import *
from unifold.data import utils
from unifold.data.lmdb_dataset import LMDBDataset
from unifold.data.data_ops import NumpyDict, TorchDict
from unifold.data.process import process_features, process_labels
from unifold.data.process_multimer import (
    pair_and_merge,
    add_assembly_features,
    convert_monomer_features,
    post_process,
    merge_msas,
)

from unicore.data import UnicoreDataset, data_utils
from unicore.distributed import utils as distributed_utils

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]


import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features
    if cfg.common.is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def process_label(all_atom_positions: np.ndarray, operation: Operation) -> np.ndarray:
    if operation == "I":
        return all_atom_positions
    rot, trans = operation
    rot = np.array(rot).reshape(3, 3)
    trans = np.array(trans).reshape(3)
    return all_atom_positions @ rot.T + trans


def get_datetime(date):
    return datetime.strptime(date, "%Y-%m-%d")


pdb_release_date = None


def filter_templates_by_date(
    template_feature,
    train_max_date,
):
    global pdb_release_date
    if pdb_release_date is None:
        return template_feature
    keep_indices = []
    max_date = get_datetime(train_max_date)
    for i, t_name in enumerate(template_feature["template_domain_names"]):
        pdb_id = t_name.decode("ascii").split("_")[0]
        if pdb_id not in pdb_release_date:
            continue
        if get_datetime(pdb_release_date[pdb_id][0]) > max_date:
            continue
        keep_indices.append(i)
    new_template_feature = {}
    for key in template_feature:
        new_template_feature[key] = template_feature[key][keep_indices]
    return new_template_feature


@utils.lru_cache(maxsize=8, copy=True)
def load_single_feature(
    sequence_id: str,
    feature_dir,
    msa_feature_dir: str,
    template_feature_dir: str,
    uniprot_msa_feature_dir: Optional[str] = None,
    is_monomer: bool = False,
    train_max_date: Optional[str] = None,
) -> NumpyDict:

    if not isinstance(feature_dir, LMDBDataset):
        monomer_feature = utils.load_pickle(
            os.path.join(feature_dir, f"{sequence_id}.feature.pkl.gz")
        )
    else:
        # lmdb dataset
        monomer_feature = feature_dir.get_by_key(sequence_id)
    msa_feature = utils.load_pickle(
        os.path.join(msa_feature_dir, f"{sequence_id}.msa.pkl.gz")
    )

    template_feature = utils.load_pickle(
        os.path.join(template_feature_dir, f"{sequence_id}.template.pkl.gz")
    )
    template_feature = {
        k: template_feature[k]
        for k in [
            "template_aatype",
            "template_all_atom_masks",
            "template_all_atom_positions",
            "template_sum_probs",
            "template_domain_names",
        ]
    }
    template_feature = filter_templates_by_date(template_feature, train_max_date)
    monomer_feature = {**monomer_feature, **msa_feature, **template_feature}
    monomer_feature = convert_monomer_features(monomer_feature)
    chain_feature = {**monomer_feature}

    if uniprot_msa_feature_dir is not None:
        all_seq_feature = utils.load_pickle(
            os.path.join(uniprot_msa_feature_dir, f"{sequence_id}.uniprot.pkl.gz")
        )
        if is_monomer:
            chain_feature["msa"], chain_feature["deletion_matrix"] = merge_msas(
                chain_feature["msa"],
                chain_feature["deletion_matrix"],
                all_seq_feature["msa"],
                all_seq_feature["deletion_matrix"],
            )
        else:
            all_seq_feature = utils.convert_all_seq_feature(all_seq_feature)
            for key in [
                "msa_all_seq",
                "msa_species_identifiers_all_seq",
                "deletion_matrix_all_seq",
            ]:
                chain_feature[key] = all_seq_feature[key]

    return chain_feature


def load_single_label(
    label_id: str,
    label_dir: str,
    symmetry_operation: Optional[Operation] = None,
) -> NumpyDict:
    if not isinstance(label_dir, LMDBDataset):
        label = utils.load_pickle(os.path.join(label_dir, f"{label_id}.label.pkl.gz"))
    else:
        # lmdb dataset
        label = label_dir.get_by_key(label_id)

    if "resolution" not in label:
        label["resolution"] = np.array([0.0])
    if symmetry_operation is not None:
        label["all_atom_positions"] = process_label(
            label["all_atom_positions"], symmetry_operation
        )
    label = {
        k: v
        for k, v in label.items()
        if k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
    }
    return label


def load(
    sequence_ids: List[str],
    feature_dir,
    msa_feature_dir: str,
    template_feature_dir: str,
    uniprot_msa_feature_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
    train_max_date: Optional[str] = None,
) -> NumpyExample:

    all_chain_features = [
        load_single_feature(
            s,
            feature_dir,
            msa_feature_dir,
            template_feature_dir,
            uniprot_msa_feature_dir,
            is_monomer,
            train_max_date,
        )
        for s in sequence_ids
    ]

    if label_ids is not None:
        # load labels
        assert len(label_ids) == len(sequence_ids)
        assert label_dir is not None
        if symmetry_operations is None:
            symmetry_operations = ["I" for _ in label_ids]
        all_chain_labels = [
            load_single_label(l, label_dir, o)
            for l, o in zip(label_ids, symmetry_operations)
        ]
        # update labels into features to calculate spatial cropping etc.
        [f.update(l) for f, l in zip(all_chain_features, all_chain_labels)]

    all_chain_features = add_assembly_features(all_chain_features)

    # get labels back from features, as add_assembly_features may alter the order of inputs.
    if label_ids is not None:
        all_chain_labels = [
            {
                k: f[k]
                for k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
            }
            for f in all_chain_features
        ]
    else:
        all_chain_labels = None

    asym_len = np.array([c["seq_length"] for c in all_chain_features], dtype=np.int64)
    if is_monomer:
        all_chain_features = all_chain_features[0]
    else:
        all_chain_features = pair_and_merge(all_chain_features)
        all_chain_features = post_process(all_chain_features)
    all_chain_features["asym_len"] = asym_len

    return all_chain_features, all_chain_labels


def process(
    config: mlc.ConfigDict,
    mode: str,
    features: NumpyDict,
    labels: Optional[List[NumpyDict]] = None,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
) -> TorchExample:

    if mode == "train":
        assert batch_idx is not None
        with data_utils.numpy_seed(seed, batch_idx, key="recycling"):
            num_iters = np.random.randint(0, config.common.max_recycling_iters + 1)
            use_clamped_fape = np.random.rand() < config[mode].use_clamped_fape_prob
    else:
        num_iters = config.common.max_recycling_iters
        use_clamped_fape = 1

    features["num_recycling_iters"] = int(num_iters)
    features["use_clamped_fape"] = int(use_clamped_fape)
    features["is_distillation"] = int(is_distillation)
    if is_distillation and "msa_chains" in features:
        features.pop("msa_chains")

    num_res = int(features["seq_length"])
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)

    if labels is not None:
        features["resolution"] = labels[0]["resolution"].reshape(-1)

    with data_utils.numpy_seed(seed, data_idx, key="protein_feature"):
        features["crop_and_fix_size_seed"] = np.random.randint(0, 63355)
        features = utils.filter(features, desired_keys=feature_names)
        features = {k: torch.tensor(v) for k, v in features.items()}
        with torch.no_grad():
            features = process_features(features, cfg.common, cfg[mode])

    if labels is not None:
        labels = [{k: torch.tensor(v) for k, v in l.items()} for l in labels]
        with torch.no_grad():
            labels = process_labels(labels)

    return features, labels


def load_and_process(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    **load_kwargs,
):
    try:
        is_monomer = (
            is_distillation
            if "is_monomer" not in load_kwargs
            else load_kwargs.pop("is_monomer")
        )
        features, labels = load(
            **load_kwargs,
            is_monomer=is_monomer,
            train_max_date=config.common.train_max_date,
        )
        features, labels = process(
            config, mode, features, labels, seed, batch_idx, data_idx, is_distillation
        )
        return features, labels
    except Exception as e:
        print("Error loading data", load_kwargs, e)
        raise e


def load_json(filename):
    return json.load(open(filename, "r"))


def get_seq_sample_weight(len, cs):
    p1 = max(min(len, 512), 256) / 512
    p2 = len**2 / 1024
    cs = max(cs, 1)
    return min(p1, p2) / cs


def load_folders(path, mode="traineval"):
    feature_path = LMDBDataset(os.path.join(path, mode, "features.lmdb"))
    msa_feature_path = os.path.join(path, mode, "msa_features")
    template_feature_path = os.path.join(path, mode, "template_features")
    label_path = LMDBDataset(os.path.join(path, mode, "labels.lmdb"))
    return feature_path, msa_feature_path, template_feature_path, label_path


class UnifoldDataset(UnicoreDataset):
    def __init__(
        self,
        args,
        seed,
        config,
        data_path,
        mode="train",
        max_step=None,
        disable_sd=False,
        json_prefix="",
    ):
        self.path = data_path
        global pdb_release_date
        pdb_release_date = load_json(
            os.path.join(self.path, "traineval", json_prefix + "release_date.json")
        )
        self.multi_label = load_json(
            os.path.join(
                self.path, "traineval", json_prefix + mode + "_multi_label.json"
            )
        )
        max_date = get_datetime(config.data.common.train_max_date)

        def filter_chain_by_date(multi_label, pdb_release_date, max_date):
            if max_date is None:
                return multi_label
            filtered_multi_label = {}
            filter_cnt = 0
            for seq in multi_label:
                cur_list = []
                for chain in multi_label[seq]:
                    pdb_id = chain.split("_")[0]
                    if get_datetime(pdb_release_date[pdb_id][0]) <= max_date:
                        cur_list.append(chain)
                    else:
                        filter_cnt += 1
                if len(cur_list) > 0:
                    filtered_multi_label[seq] = cur_list
            logger.info(
                "Filter out %d chains with release date after %s",
                filter_cnt,
                max_date,
            )
            return filtered_multi_label

        if mode == "train":
            self.multi_label = filter_chain_by_date(
                self.multi_label, pdb_release_date, max_date
            )

        self.inverse_multi_label = self._inverse_map(self.multi_label)

        def load_sample_weight(multi_label, dir, mode):
            cluster_size = load_json(
                os.path.join(self.path, dir, json_prefix + mode + "_cluster_size.json")
            )
            seq_length = load_json(
                os.path.join(self.path, dir, json_prefix + mode + "_seq_length.json")
            )
            sample_weight = {}
            keys = multi_label.keys() if multi_label is not None else seq_length.keys()
            for seq in keys:
                sample_weight[seq] = get_seq_sample_weight(
                    seq_length[seq], cluster_size[seq]
                )
            return sample_weight

        self.seq_sample_weight = load_sample_weight(self.multi_label, "traineval", mode)
        logger.info(
            "load {} chains (unique {} sequences)".format(
                len(self.inverse_multi_label), len(self.seq_sample_weight)
            )
        )
        (
            self.feature_path,
            self.msa_feature_path,
            self.template_feature_path,
            self.label_path,
        ) = load_folders(self.path, mode="traineval")
        if mode == "train" and not disable_sd:
            self.sd_sample_weight = load_sample_weight(None, "sd", "sd")
            logger.info(
                "load {} self-distillation samples.".format(len(self.sd_sample_weight))
            )
            (
                self.sd_feature_path,
                self.sd_msa_feature_path,
                self.sd_template_feature_path,
                self.sd_label_path,
            ) = load_folders(self.path, mode="sd")
        else:
            self.sd_sample_weight = None
        self.batch_size = (
            args.batch_size
            * distributed_utils.get_data_parallel_world_size()
            * args.update_freq[0]
        )
        if mode == "train":
            self.data_len = max_step * self.batch_size
        else:
            self.data_len = max_step
        self.mode = mode
        self.num_seq, self.seq_keys, self.seq_sample_prob = self.cal_sample_weight(
            self.seq_sample_weight
        )
        if self.sd_sample_weight is not None:
            (
                self.sd_num_chain,
                self.sd_chain_keys,
                self.sd_sample_prob,
            ) = self.cal_sample_weight(self.sd_sample_weight)
        self.config = config.data
        self.seed = seed
        self.sd_prob = args.sd_prob

    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    def sample_seq(self, idx):
        is_distillation = False
        if self.mode == "train":
            with data_utils.numpy_seed(self.seed, idx, key="data_sample"):
                is_distillation = (
                    (np.random.rand(1)[0] < self.sd_prob)
                    if self.sd_sample_weight is not None
                    else False
                )
                if is_distillation:
                    prot_idx = np.random.choice(
                        self.sd_num_chain, p=self.sd_sample_prob
                    )
                    label_name = self.sd_chain_keys[prot_idx]
                    seq_name = label_name
                else:
                    seq_idx = np.random.choice(self.num_seq, p=self.seq_sample_prob)
                    seq_name = self.seq_keys[seq_idx]
                    label_name = np.random.choice(self.multi_label[seq_name])
        else:
            seq_idx = idx % self.num_seq
            seq_name = self.seq_keys[seq_idx]
            label_name = np.random.choice(self.multi_label[seq_name])
        return seq_name, label_name, is_distillation

    def __getitem__(self, idx):
        sequence_id, label_id, is_distillation = self.sample_seq(idx)
        feature_path, msa_feature_path, template_feature_path, label_path = (
            (
                self.feature_path,
                self.msa_feature_path,
                self.template_feature_path,
                self.label_path,
            )
            if not is_distillation
            else (
                self.sd_feature_path,
                self.sd_msa_feature_path,
                self.sd_template_feature_path,
                self.sd_label_path,
            )
        )
        features, _ = load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=[sequence_id],
            feature_dir=feature_path,
            msa_feature_dir=msa_feature_path,
            template_feature_dir=template_feature_path,
            uniprot_msa_feature_dir=None,
            label_ids=[label_id],
            label_dir=label_path,
            symmetry_operations=None,
            is_monomer=True,
        )
        return features

    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        return data_utils.collate_dict(samples, dim=1)

    @staticmethod
    def _inverse_map(mapping: Dict[str, List[str]]):
        inverse_mapping = {}
        for ent, refs in mapping.items():
            for ref in refs:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    ent_2 = inverse_mapping[ref]
                    assert (
                        ent == ent_2
                    ), f"multiple entities ({ent_2}, {ent}) exist for reference {ref}."
                inverse_mapping[ref] = ent
        return inverse_mapping


def get_chain_sample_weight(cs):
    cs = max(cs, 1)
    return 1.0 / cs


class UnifoldMultimerDataset(UnifoldDataset):
    def __init__(
        self,
        args: mlc.ConfigDict,
        seed: int,
        config: mlc.ConfigDict,
        data_path: str,
        mode: str = "train",
        max_step: Optional[int] = None,
        disable_sd: bool = False,
        json_prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            args, seed, config, data_path, mode, max_step, disable_sd, json_prefix
        )
        self.data_path = data_path
        self.pdb_assembly = json.load(
            open(
                os.path.join(
                    self.data_path,
                    "traineval",
                    json_prefix + mode + "_mmcif_assembly.json",
                )
            )
        )
        self.pdb_chains = self.get_chains(self.inverse_multi_label)
        self.uniprot_msa_feature_path = os.path.join(
            self.data_path, "traineval", "uniprot_features"
        )
        self.max_chains = args.max_chains

        def filter_pdb_assembly(pdb_assembly):
            new_pdb_assembly = {}
            for pdb_id in pdb_assembly:
                content = pdb_assembly[pdb_id]
                new_content = {"chains": [], "opers": []}
                has_content = False
                for i, chain in enumerate(content["chains"]):
                    if (pdb_id + "_" + chain) in self.inverse_multi_label:
                        new_content["chains"].append(chain)
                        new_content["opers"].append(content["opers"][i])
                        has_content = True
                if has_content:
                    new_pdb_assembly[pdb_id] = new_content
            return new_pdb_assembly

        self.pdb_assembly = filter_pdb_assembly(self.pdb_assembly)

        def load_chain_cluster_size(mode):
            cluster_size = load_json(
                os.path.join(
                    self.path, "traineval", json_prefix + mode + "_cluster_size.json"
                )
            )
            seq_cnt = {}
            for pdb_id in self.pdb_assembly:
                for chain in self.pdb_assembly[pdb_id]["chains"]:
                    seq = self.inverse_multi_label[pdb_id + "_" + chain]
                    if seq not in seq_cnt:
                        seq_cnt[seq] = 0
                    seq_cnt[seq] += 1
            new_cluster_size = {}
            for seq in seq_cnt:
                assert seq in cluster_size, seq
                assert seq in seq_cnt, seq
                new_cluster_size[seq] = cluster_size[seq] * seq_cnt[seq]
            return new_cluster_size

        chain_cluster_size = load_chain_cluster_size(mode)

        def cal_pdb_sample_weight(mode, pdb_assembly, cluster_size):
            seq_length = load_json(
                os.path.join(
                    self.path, "traineval", json_prefix + mode + "_seq_length.json"
                )
            )
            sample_weight = {}
            total_seq_length = {}
            for pdb_id in pdb_assembly:
                cur_sample_weight = 0.0
                cur_seq_length = 0
                for chain in pdb_assembly[pdb_id]["chains"]:
                    seq = self.inverse_multi_label[pdb_id + "_" + chain]
                    cur_sample_weight += get_chain_sample_weight(cluster_size[seq])
                    cur_seq_length += seq_length[seq]
                # avoid too large sample weights
                sample_weight[pdb_id] = min(cur_sample_weight, 2.0)
                total_seq_length[pdb_id] = cur_seq_length
            return (sample_weight, total_seq_length)

        self.sample_weight, total_seq_length = cal_pdb_sample_weight(
            mode, self.pdb_assembly, chain_cluster_size
        )
        self.pdb_assembly, self.sample_weight = self.filter_pdb_by_max_chains(
            self.pdb_assembly, self.sample_weight, self.max_chains, total_seq_length
        )
        self.num_pdb, self.pdb_keys, self.sample_prob = self.cal_sample_weight(
            self.sample_weight
        )

    def sample_pdb(self, idx):
        is_distillation = False
        if self.mode == "train":
            with data_utils.numpy_seed(self.seed, idx, key="data_sample"):
                is_distillation = (
                    (np.random.rand(1)[0] < self.sd_prob)
                    if self.sd_sample_weight is not None
                    else False
                )
                if is_distillation:
                    prot_idx = np.random.choice(
                        self.sd_num_chain, p=self.sd_sample_prob
                    )
                    label_name = self.sd_chain_keys[prot_idx]
                else:
                    prot_idx = np.random.choice(self.num_pdb, p=self.sample_prob)
                    label_name = self.pdb_keys[prot_idx]
        else:
            prot_idx = idx % self.num_pdb
            label_name = self.pdb_keys[prot_idx]
        return label_name, is_distillation

    def __getitem__(self, idx):
        label_id, is_distillation = self.sample_pdb(idx)
        if is_distillation:
            label_ids = [label_id]
            sequence_ids = [label_id]
            feature_path, msa_feature_path, template_feature_path, label_path = (
                self.sd_feature_path,
                self.sd_msa_feature_path,
                self.sd_template_feature_path,
                self.sd_label_path,
            )
            uniprot_msa_feature_path = None
            symmetry_operations = None
        else:
            pdb_id = label_id
            label_ids = [
                pdb_id + "_" + id for id in self.pdb_assembly[pdb_id]["chains"]
            ]
            symmetry_operations = [t for t in self.pdb_assembly[pdb_id]["opers"]]
            sequence_ids = [
                self.inverse_multi_label[chain_id] for chain_id in label_ids
            ]
            feature_path, msa_feature_path, template_feature_path, label_path = (
                self.feature_path,
                self.msa_feature_path,
                self.template_feature_path,
                self.label_path,
            )
            uniprot_msa_feature_path = self.uniprot_msa_feature_path

        return load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=sequence_ids,
            feature_dir=feature_path,
            msa_feature_dir=msa_feature_path,
            template_feature_dir=template_feature_path,
            uniprot_msa_feature_dir=uniprot_msa_feature_path,
            label_ids=label_ids,
            label_dir=label_path,
            symmetry_operations=symmetry_operations,
            is_monomer=False,
        )

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        if len(samples) <= 0:  # tackle empty batch
            return None
        feats = [s[0] for s in samples]
        labs = [s[1] for s in samples if s[1] is not None]
        try:
            feats = data_utils.collate_dict(feats, dim=1)
        except:
            raise ValueError("cannot collate features", feats)
        if not labs:
            labs = None
        return feats, labs

    @staticmethod
    def get_pdb_name(chain):
        return chain.split("_")[0]

    @staticmethod
    def get_chains(canon_chain_map):
        pdb_chains = {}
        for chain in canon_chain_map:
            pdb = UnifoldMultimerDataset.get_pdb_name(chain)
            if pdb not in pdb_chains:
                pdb_chains[pdb] = []
            pdb_chains[pdb].append(chain)
        return pdb_chains

    @staticmethod
    def filter_pdb_by_max_chains(
        pdb_assembly, sample_weight, max_chains, total_seq_length
    ):
        new_pdb_assembly = {}
        for pdb_id in pdb_assembly:
            size = len(pdb_assembly[pdb_id]["chains"])
            if size <= max_chains:
                new_pdb_assembly[pdb_id] = pdb_assembly[pdb_id]
        new_sample_weight = {
            k: sample_weight[k]
            for k in sample_weight
            if UnifoldMultimerDataset.get_pdb_name(k) in new_pdb_assembly
        }
        logger.info(
            f"filtered out {len(pdb_assembly) - len(new_pdb_assembly)} / {len(pdb_assembly)} PDBs "
            f"by max_chains {max_chains}"
        )
        return new_pdb_assembly, new_sample_weight
