from unifold.dataset import *
from unifold.data.process import process_features_single
import gzip
import pickle


@utils.lru_cache(maxsize=8, copy=True)
def load_emb(
    sequence_id: str,
    monomer_feature_dir: str,
    emb_dir: str,
) -> NumpyDict:

    monomer_feature = utils.load_pickle(
        os.path.join(monomer_feature_dir, f"{sequence_id}.feature.pkl.gz")
    )
    
    chain_feature = {}
    chain_feature["aatype"] = np.argmax(monomer_feature["aatype"], axis=-1).astype(
        np.int32
    )
    chain_feature["sequence"] = monomer_feature["sequence"]
    # monomer_feature = utils.load_pickle(
    #     os.path.join(emb_dir, f"{sequence_id}.esm2_emb.pkl.gz")
    # )
    # token = monomer_feature["token"]
    # chain_feature["token"] = token
    # chain_feature["pair"] = np.transpose(monomer_feature["pair"], [1, 2, 0])
    chain_feature["seq_length"] = np.array([len(chain_feature["aatype"])])
    chain_feature["residue_index"] = np.arange(0, len(chain_feature["aatype"]))
    # assert (
    #     chain_feature["pair"].shape[0]
    #     == chain_feature["pair"].shape[1]
    #     == chain_feature["token"].shape[0]
    #     == chain_feature["aatype"].shape[0]
    # )
    return chain_feature


def merge_multi_emb(all_chain_features):
    merge_features = {}
    num_chains = len(all_chain_features)
    for key in all_chain_features[0]:
        if key not in ["sequence", "resolution", "pair", "seq_length"]:
            merge_features[key] = np.concatenate(
                [x[key] for x in all_chain_features], axis=0
            )
    # total_length = sum(x["aatype"].shape[0] for x in all_chain_features)
    # pair = dok_matrix(
    #     (total_length, total_length, all_chain_features[0]["pair"].shape[-1]),
    #     dtype=all_chain_features[0]["pair"].dtype,
    # )
    # offset = 0
    # for x in all_chain_features:
    #     cur_len = x["aatype"].shape[0]
    #     pair[offset : offset + cur_len, offset : offset + cur_len, :] = x["pair"]
    #     offset += cur_len
    # merge_features["pair"] = pair
    merge_features["seq_length"] = np.asarray(
        merge_features["aatype"].shape[0], dtype=np.int32
    )
    return merge_features


def load_crop_emb(
    features, asymid_2_seq, per_asym_residue_index, emb_dir, mode="train"
):
    total_len = features["aatype"].shape[-1]
    all_pair = None
    all_token = None
    offset = 0
    for asym_id in per_asym_residue_index:
        crop_idx = per_asym_residue_index[asym_id]
        seq = asymid_2_seq[asym_id]
        emb_feature = utils.load_pickle(
            os.path.join(emb_dir, f"{seq}.esm2_multimer_finetune_emb.pkl.gz")
        )
        token = torch.from_numpy(emb_feature["token"])
        pair = torch.from_numpy(emb_feature["pair"])
        if all_token is None:
            all_token = token.new_zeros(total_len, token.shape[-1])
        if all_pair is None:
            all_pair = pair.new_zeros(total_len, total_len, pair.shape[-3])
        if mode != "predict":
            token = torch.index_select(token, 0, crop_idx)
            pair = torch.index_select(pair, -1, crop_idx)
            pair = torch.index_select(pair, -2, crop_idx)
        pair = pair.permute(1, 2, 0)
        cur_len = token.shape[0]
        all_token[offset : offset + cur_len, :] = token
        all_pair[offset : offset + cur_len, offset : offset + cur_len, :] = pair
        offset += cur_len

    features["token"] = all_token[None, ...]
    features["pair"] = all_pair[None, ...]
    return features


def load_assembly_esm(embdir, assem_name):
    fn = os.path.join(embdir, f"{assem_name}.esm2_multimer_finetune_emb.pkl.gz")
    if not os.path.exists(fn):
        fn = os.path.join(embdir, f"{assem_name}.esm2_3b_emb.pkl.gz")
    if not os.path.exists(fn):
        fn = os.path.join(embdir, f"{assem_name}.esm2_15b_emb.pkl.gz")
    with gzip.GzipFile(fn, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings["token"]


def load(
    sequence_ids: List[str],
    feature_dir,
    msa_feature_dir: str,
    emb_dir: str,
    template_feature_dir: str,
    uniprot_msa_feature_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
    train_max_date: Optional[str] = None,
    is_distillation=False,
) -> NumpyExample:

    if is_distillation:
        assemb_name = sequence_ids[0]
    else:
        try:
            assemb_name = label_ids[0].split("_")[0]
        except:
            assemb_name = sequence_ids[0].split("_")[0]
    embeddings = load_assembly_esm(emb_dir, assemb_name)

    all_chain_features = [load_emb(s, feature_dir, emb_dir) for s in sequence_ids]
    assert embeddings.shape[0] == sum(
        [len(chain_feature["aatype"]) for chain_feature in all_chain_features]
    ), "embedding shape error {} {} {}".format(
        str(label_ids),
        embeddings.shape[0],
        sum([len(chain_feature["aatype"]) for chain_feature in all_chain_features]),
    )
    curpos = 0
    for feat in all_chain_features:
        offset = len(feat["aatype"])
        feat["token"] = embeddings[curpos : curpos + offset]
        curpos += offset

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

    asym_len = np.array(
        [int(c["seq_length"]) for c in all_chain_features], dtype=np.int64
    )
    all_chain_features = merge_multi_emb(all_chain_features)
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
    emb_dir: str = None,
    **kwargs,
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
            features = process_features_single(features, cfg.common, cfg[mode])

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
            is_distillation=is_distillation,
        )
        features, labels = process(
            config,
            mode,
            features,
            labels,
            seed,
            batch_idx,
            data_idx,
            is_distillation,
            **load_kwargs,
        )
        return features, labels
    except Exception as e:
        print("Error loading data", load_kwargs, e)
        raise e


class UnifoldSingleMultimerDataset(UnifoldMultimerDataset):
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
            args,
            seed,
            config,
            data_path,
            mode,
            max_step,
            disable_sd,
            json_prefix,
            **kwargs,
        )

        def load_sample_weight(multi_label, dir, mode):
            cluster_size = load_json(
                os.path.join(
                    self.path, dir, json_prefix + mode + "_cluster_size_all.json"
                )
            )
            seq_length = load_json(
                os.path.join(
                    self.path, dir, json_prefix + mode + "_seq_length_all.json"
                )
            )
            sample_weight = {}
            keys = multi_label.keys() if multi_label is not None else seq_length.keys()
            for seq in keys:
                sample_weight[seq] = get_seq_sample_weight(
                    seq_length[seq], cluster_size[seq]
                )
            return sample_weight

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
        if self.sd_sample_weight is not None:
            (
                self.sd_num_chain,
                self.sd_chain_keys,
                self.sd_sample_prob,
            ) = self.cal_sample_weight(self.sd_sample_weight)

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

        def filter_pdb_assembly(pdb_assembly, config):

            new_pdb_assembly = {}
            if config.data.common.feature_src:
                filter_keys = json.load(
                    open(
                        os.path.join(
                            self.data_path,
                            "traineval",
                            f"{config.data.common.feature_src}_filter",
                            json_prefix + "filtered_" + mode + "_keys.json",
                        )
                    )
                )
            else:
                filter_keys = json.load(
                    open(
                        os.path.join(
                            self.data_path,
                            "traineval",
                            json_prefix + "filtered_" + mode + "_keys.json",
                        )
                    )
                )
            for pdb_id in pdb_assembly:
                if pdb_id in filter_keys:
                    # print(f"filter {pdb_id} too long")
                    continue
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

        self.pdb_assembly = filter_pdb_assembly(self.pdb_assembly, config=config)

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
        if config.data.common.feature_src:
            self.emb_path = os.path.join(
                self.data_path, "traineval", f"esms_{config.data.common.feature_src}"
            )
            self.sd_emb_path = os.path.join(
                self.data_path, "sd", f"esms_{config.data.common.feature_src}"
            )
        else:
            self.emb_path = os.path.join(self.data_path, "traineval", "esms")
            self.sd_emb_path = os.path.join(self.data_path, "sd", "esms")

    def __getitem__(self, idx):
        label_id, is_distillation = self.sample_pdb(idx)
        if is_distillation:
            label_ids = [label_id]
            sequence_ids = [label_id]
            monomer_feature_path, label_path, emb_path = (
                self.sd_feature_path,
                self.sd_label_path,
                self.sd_emb_path,
            )
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
            monomer_feature_path, label_path, emb_path = (
                self.feature_path,
                self.label_path,
                self.emb_path,
            )

        return load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=sequence_ids,
            feature_dir=monomer_feature_path,
            msa_feature_dir=None,
            template_feature_dir=None,
            uniprot_msa_feature_dir=None,
            emb_dir=emb_path,
            label_ids=label_ids,
            label_dir=label_path,
            symmetry_operations=symmetry_operations,
            is_monomer=False,
        )
