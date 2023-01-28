
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


Attention mask & Attention bias shape support
------------

```
Support the shape of q/k/v as follow:
q's shape [total_size * head, seq_q, head_dim]
k's shape [total_size * head, seq_k, head_dim]
v's shape [total_size * head, seq_k, head_dim]

Attention Mask 
[total_size, head, seq_q, seq_k]
1. total_size must be the same with q's total_size
2. head must be 1 or head like shape in q
3. seq_q must be 1  
4. seq_k must be the same with k's seq_k 


Attention Bias
[total_size, head, seq_q, seq_k]
1. total_size must be 1
2. head must be the same with q's head
3. seq_q must be the same with q's seq_q
4. seq_k must be the same with k's seq_k
```

If you need more different shape size support, any contribution or discussion is welcome. 
