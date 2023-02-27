from unifold.modules.auxillary_heads import *


class AuxiliaryHeadsSingle(AuxiliaryHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked_msa = None
