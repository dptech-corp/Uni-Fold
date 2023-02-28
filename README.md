# Uni-Fold MuSSe: De Novo Protein Complex Prediction with Protein Language Models
## Installation and Preparations
### Installing Uni-Fold MuSSe

Our code is implemented on a distributed PyTorch framework [Uni-Core](https://github.com/dptech-corp/Uni-Core#installation), and for convenience we also provide a Docker image.
```bash
docker pull dptechnology/unifold:latest-pytorch1.11.0-cuda11.3
```
Then, you can create and attach into the docker container, and clone & install unifold.
```shell
git clone --single-branch -b unifold_musse git@github.com:dptech-corp/Uni-Fold.git unifold_musse
cd unifold_musse
pip install -e .
```
### Downloading the pre-trained model parameters
Use the following command to download the parameters of our further pre-trained protein language model and single sequence protein complex predictor:
```shell
# the protein language model
wget https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unifold_model/unifold_musse/plm.pt 

# the protein complex predictor
wget https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unifold_model/unifold_musse/mp.pt
```

### Running Uni-Fold MuSSe
Run the following command to predict the structure of the target fasta:
```shell
bash run_unifold_musse.sh \
    /path/to/the/input.fasta \   # target fasta file
    /path/to/the/output/directory/ \    # output directory
    /path/to/multimer_model_parameters.pt \ # multimer predictor parameters
    /path/to/pretrain_lm_parameters.pt  # language model parameters
    
```