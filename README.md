# Uni-Fold: an open-source platform for developing protein folding models beyond AlphaFold.

Uni-Fold is a thoroughly open-source platform for developing protein folding models beyond [AlphaFold](https://github.com/deepmind/alphafold/), with following advantages:

- Reimplemented AlphaFold and AlphaFold-Multimer models in PyTorch framework. The first project for AlphaFold-Multimer training.

- Model correctness proved by successful from-scratch training with equivalent accuracy.

- Highest efficiency among existing AlphaFold implementations.

- Rich features from [Uni-Core](https://github.com/dptech-corp/Uni-Core/), such as efficient fp16/bf16 training, per sample gradient clipping, and fused kernels. 

The name Uni-Fold is inherited from Uni-Fold-JAX. First released on Dec 8 2021, [Uni-Fold-JAX](https://github.com/dptech-corp/Uni-Fold-jax) was the first open-source project (with training scripts) that successfully reproduced the from-scratch training of AlphaFold. Until now, Uni-Fold-JAX is still the only project that supports training of the original AlphaFold implementation in JAX framework. Due to efficiency and collaboration considerations, we moved from Jax to PyTorch on Jan 2022, based on which we further developed the multimer models.


## Installation

[Instructions on installation]


## Preparing the datasets

Training and inference with Uni-Fold require homology searches on sequence and structure databases. Use the following command to download these databases:

```bash
  bash scripts/download_all_data.sh /path/to/database/directory
```

Make sure there is at least 3TB storage space for downloading (~500GB) and uncompressing the databases.


## Downloading the pre-trained model parameters

Parameters are coming soon :)

<!-- Inferenece and finetuning with Uni-Fold requires pretrained model parameters. Use the following command to download the parameters: -->

## Converting the AlphaFold and OpenFold parameters to Uni-Fold

[converting scripts]

## Running Uni-Fold

After properly configurating the environment and databases, run the following command to predict the structure of the target fasta:

```bash
bash run_unifold.sh \
    /path/to/the/input.fasta \        # target fasta file
    /path/to/the/output/directory/ \  # output directory
    /path/to/database/directory/ \    # directory of databases
    2020-05-01 \                      # use templates before this date
    model_2_af2 \                     # specify model name
    /path/to/model_parameters.pt      # model parameters
```

For monomer prediction, each fasta file shall contain only one sequence; for multimer prediction, the input fasta file shall contain all sequences of the target complex, **with duplicated homologous sequences**.

## Uni-Fold outputs

The outputs of Uni-Fold inferences contain the predicted structures in `*.pdb` files. Besides, other outputs are dumped in `*.pkl.gz` files. We summarize the confidence metrics, namely `plddt` and `iptm+ptm` in `*.json` files.

## Training Uni-Fold

Training Uni-Fold relies on pre-calculated features of proteins. We provide a demo dataset in the [example data](example_data) folder. A larger dataset will be released soon.

### Demo case

To start with, we provide a demo script to train the monomer/multimer system of Uni-Fold:

```bash
bash train_monomer_demo.sh .
```

and

```bash
bash train_multimer_demo.sh .
```

This command starts a training process on the [demo data](example_data) included in this repository. Note that this demo script only tests the correctness of package installation and does not reflect any true performances. 


### From-scratch Training

Run the following command to train Uni-Fold Monomer/Multimer from-scratch:

```bash
bash train_monomer.sh \                 # train_multimer.sh for multimer
    /path/to/training/data/directory/ \ # dataset directory
    /path/to/output/directory/ \        # output directory where parameters are stored
    model_2_af2                         # model name
```

Notice that: 

1. The dataset directory should be configurated in a similar way as [example data](example_data).
2. The output directory should have enough space to store model parameters (~3GB per checkpoint, so empirically >100GB satisfies).
3. We provide several default model names in [config.py](unifold/config.py), namely `model_1`, `model_2`, `model_2_af2` etc. for monomer models and `multimer`, `multimer_af2` etc. for multimer models. Check `model_config()` function for the differences between model names. You may also personalize your own model by modifying the function (i.e. forking the if-elses).


### Finetuning

Run the following command to finetune Uni-Fold Monomer/Multimer pretrained models:

```bash
bash finetune_monomer.sh \              # finetune_multimer.sh for multimer
    /path/to/training/data/directory/ \ # dataset directory
    /path/to/output/directory/ \        # output directory where parameters are stored
    /path/to/pretrained/parameters.pt \ # pretrained parameters
    model_2_af2                         # model name
```

Besides the notices in the previous section, additionaly notice that: 

1. The model architecture should be correctly specified by the model name.
2. Checkpoints must be in Uni-Fold format (`*.pt`).


## Citing this work

Citation is coming soon :)

<!-- If you use the code or data in this package, please cite:

```bibtex

``` -->

## Acknowledgements

Our training framework is based on [Uni-Core](https://github.com/dptech-corp/Uni-Core/). Implementation of fused operators referred to  [fused_ops](https://github.com/guolinke/fused_ops/). We partly referred to an early version of [OpenFold](https://github.com/aqlaboratory/openfold) for some of the PyTorch implementation, while mostly followed the original code of [AlphaFold](https://github.com/deepmind/alphafold/). For the data processing part, we followed [AlphaFold](https://github.com/deepmind/alphafold/), and referred to utilities in [Biopython](https://biopython.org/), [HH-suite3](https://github.com/soedinglab/hh-suite/), [HMMER](http://eddylab.org/software/hmmer/), [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), and [SciPy](https://scipy.org/).

## License and Disclaimer

Copyright 2022 DP Technology.

### Uni-Fold Code License

Uni-Fold is licensed under permissive Apache Licence, Version 2.0.

### Model Parameters License

The Uni-Fold parameters are made available under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You can find details at: https://creativecommons.org/licenses/by/4.0/legalcode

### Contributing to Uni-Fold

Uni-Fold is an ongoing project. Our target is to develop better protein folding models and to apply them in real scenarios together with the entire community. We welcome all contributions to this repository, including but not limited to 1) reports and fixes of bugs, 2) new features and 3) accuracy and efficiency improvements. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
