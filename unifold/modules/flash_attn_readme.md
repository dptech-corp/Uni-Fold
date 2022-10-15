
Installation
------------

**Build from source**

```
git clone https://github.com/dptech-corp/flash-attention.git
cd flash-attention
python setup.py install

# or multi-core compiling accleartion

python setup.py build -j 8 install
```

**Use pre-compiled python wheels**

We also pre-compiled wheels by GitHub Actions. You can download them from the [Release](https://github.com/dptech-corp/flash-attention/releases/). And you should check the pyhon version, PyTorch version and CUDA version. For example, for PyToch 1.12.1, python 3.7, and CUDA 11.3, you can install [flash_attn-0.1+cu113torch1.12.1-cp37-cp37m-linux_x86_64.whl](https://github.com/dptech-corp/flash-attention/releases/download/refs%2Fheads%2Fworkflow/flash_attn-0.1+cu113torch1.12.1-cp37-cp37m-linux_x86_64.whl). 




Use Flash-Attention in Uni-Fold
------------

Changeing the configuration `use_flash_attn` to `True` in `unifold/config.py`, you will use Flash-Attention acceleration for the Uni-Fold.



Notice
------------
Be careful the the odd length of attention mask and attention bias is not support now due to the instruction, we will support in the later.