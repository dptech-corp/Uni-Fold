from pathlib import Path
import importlib
from .msa_dataset import MultimerDataset, PPIDataset, SingleDataset, MSAConcatDataset
from .bert_tokenize_dataset import BertTruncateDataset, BertDataset
from .mask_tokens_dataset import MaskTokensDataset
from .position_dataset import IsSameEntityDataset, HasSameSequenceDataset
from .position_pad_dataset import PositionRightPadDataset

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("plm.data." + file.name[:-3])