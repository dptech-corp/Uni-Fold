import os
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool
import glob
import pickle
import numpy as np
from functools import partial
import gzip

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",  # Also U.
    2: "D",  # Also B.
    3: "E",  # Also Z.
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  # Includes J and O.
    21: "-",
}

restypes_with_x_and_gap = restypes + ["X", "-"]
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i])
    for i in range(len(restypes_with_x_and_gap))
)

MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = np.array(
    MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, dtype=np.int8
)

data_dir = sys.argv[1]
data_type = sys.argv[2]
output_dir = sys.argv[3]
prefix = sys.argv[4]


feature_dir = "{}/{}_features/".format(data_dir, data_type)
label_dir = "{}/{}_labels/".format(data_dir, data_type)

feature_files = glob.glob(feature_dir + "*")
cluster_size = json.load(
    open(os.path.join(data_dir, "{}_cluster_size.json".format(data_type)))
)
if data_type == "pdb":
    multi_label = json.load(open(os.path.join(data_dir, "pdb_multi_label.json")))
else:
    multi_label = None

new_sample_weight = {}
new_multi_label = {}


def __load_from_file__(path):
    if path.endswith(".pkl"):
        return pickle.load(open(path, "rb"))
    else:
        return pickle.load(gzip.open(path, "rb"))


def get_sample_weight(len, cs):
    p1 = max(min(len, 512), 256) / 512
    p2 = len**2 / 1024
    return min(p1, p2) / cs


def check_one_file(file):
    t = os.path.split(file)[-1].split(".")[0]
    raw_feature = __load_from_file__(file)
    seq_len = raw_feature["aatype"].shape[0]
    aatype = np.argmax(raw_feature["aatype"], axis=-1)
    msa_aatype = np.array(
        [MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[ii] for ii in raw_feature["msa"][0]]
    )
    if not (aatype == msa_aatype).all():
        return t, None, None

    _, counts = np.unique(aatype, return_counts=True)
    freqs = counts.astype(np.float32) / seq_len
    max_freq = np.max(freqs)
    labels = []

    def load_and_check_label(label_t, seq_len):
        label_filename = os.path.join(label_dir, "{}.label.pkl.gz".format(label_t))
        if os.path.isfile(label_filename):
            raw_label = __load_from_file__(label_filename)
            label_aatype = raw_label["aatype_index"]
            label_seq_len = raw_label["all_atom_positions"].shape[0]
            resolution = raw_label["resolution"].reshape(1)[0]
            if (
                label_seq_len == seq_len
                and (aatype == label_aatype).all()
                and resolution < 9
            ):
                return True
        return False

    if multi_label is None or t not in multi_label:
        if load_and_check_label(t, seq_len):
            labels.append(t)
    else:
        for label_t in multi_label[t]:
            if load_and_check_label(label_t, seq_len):
                labels.append(label_t)
    if len(labels) > 0 and t in cluster_size and max_freq < 0.8:
        sample_weight = get_sample_weight(seq_len, cluster_size[t])
        return t, sample_weight, labels
    else:
        return t, None, None


file_cnt = len(feature_files)
filter_cnt = 0
error_features = []
error_labels = []
with Pool(96) as pool:
    for ret in tqdm(pool.imap(check_one_file, feature_files), total=file_cnt):
        t, sw, ll = ret
        if sw is not None:
            new_sample_weight[t] = sw
            new_multi_label[t] = ll
            if multi_label is not None and len(ll) < len(multi_label[t]):
                for x in multi_label[t]:
                    if x not in ll:
                        error_labels.append(x)
            else:
                if len(ll) <= 0:
                    error_labels.append(t)
        else:
            error_features.append(t)
            filter_cnt += 1

print(len(error_features), len(error_labels))


def write_list_to_file(a, file):
    with open(file, "w") as output:
        for x in a:
            output.write(str(x) + "\n")


write_list_to_file(
    error_features, "{}/{}_error_features.txt".format(output_dir, data_type)
)
write_list_to_file(error_labels, "{}/{}_error_labels.txt".format(output_dir, data_type))

if data_type == "pdb":
    json.dump(
        new_sample_weight,
        open("{}/{}_train_sample_weight.json".format(output_dir, prefix), "w"),
        indent=4,
    )
    json.dump(
        new_multi_label,
        open("{}/{}_train_multi_label.json".format(output_dir, prefix), "w"),
        indent=4,
    )
else:
    json.dump(
        new_sample_weight,
        open("{}/{}_sd_train_sample_weight.json".format(output_dir, prefix), "w"),
        indent=4,
    )
