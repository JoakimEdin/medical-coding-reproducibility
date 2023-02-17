# ⚕️Automated Medical Coding

## Introduction 
Automatic medical coding is the task of automatically assigning diagnosis and procedure codes based on discharge summaries from electronic health records. This repository contains code for easily implementing new automatic medical coding machine learning models. Furthermore, the repository contains new splits for MIMIC-III and the newly released MIMIC-IV. The following models have been implemented:

| Model | Paper | Original Code |
| ----- | ----- | ------------- |
| CNN   |[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
| Bi-GRU|[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
|CAML   |[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
| MultiResCNN | [ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network](https://arxiv.org/pdf/1912.00862.pdf) | [link](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network) |
| LAAT | [A Label Attention Model for ICD Coding from Clinical Text](https://arxiv.org/abs/2007.06351) | [link](https://github.com/aehrc/LAAT) |
| PLM-ICD | [PLM-ICD: Automatic ICD Coding with Pretrained Language Models](https://aclanthology.org/2022.clinicalnlp-1.2/) | [link](https://github.com/MiuLab/PLM-ICD) |


## How to reproduce results
### Setup Conda environment
1. Create a conda environement `conda create -n coding python=3.10`
2. Install the packages `pip install . -e`

### Prepare MIMIC-III
This code has been developed on MIMIC-III v1.4. 
1. Download the MIMIC-III data into your preferred location `path/to/mimiciii`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours.  - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Open the file `src/settings.py`
3. Change the variable `DOWNLOAD_DIRECTORY_MIMICIII` to the path of your downloaded data `path/to/mimiciii`
4. If you want to use the MIMIC-III full and MIMIC-III 50 from the [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) you need to run `python prepare_data/prepare_mimiciii_mullenbach.py`
5. If you want to use MIMIC-III clean from our paper you need to run `python prepare_data/prepare_mimiciii.py`

### Prepare MIMIC-IV
This code has been developed on MIMIC-IV and MIMIC-IV v2.2. 
1. Download MIMIC-IV and MIMIC-IV-NOTE into your preferred location `path/to/mimiciv` and `path/to/mimiciv-note`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours.  - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Open the file `src/settings.py`
3. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV` to the path of your downloaded data `path/to/mimiciv`
4. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV_NOTE` to the path of your downloaded data `path/to/mimiciv-note`
5. Run `python prepare_data/prepare_mimiciv.py`

### Before running experiments
1. Create a weights and biases account. It is possible to run the experiments without wandb.
2. Download the [model checkpoints](https://drive.google.com/file/d/1hYeJhztAd-JbhqHojY7ZpLtkBcthD8AK/view?usp=share_link) and unzip it.
3. If you want to train PLM-ICD, you need to download [RoBERTa-base-PM-M3-Voc](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz), unzip it and change the `model_path` parameter in `configs/model/plm_icd.yaml` to the path of the download.

### Running experiments
#### Training
You can run any experiment found in `configs/experiment`. Here are some examples:
   * Train PLM-ICD on MIMIC-III clean on GPU 0: `python main.py experiment=mimiciii_clean/plm_icd gpu=0`
   * Train CAML on MIMIC-III full on GPU 6: `python main.py experiment=mimiciii_full/caml gpu=6`
   * Train LAAT on MIMIC-IV ICD-9 full on GPU 6: `python main.py experiment=mimiciv_icd9/laat gpu=6`
   * Train LAAT on MIMIC-IV ICD-9 full on GPU 6 without weights and biases: `python main.py experiment=mimiciv_icd9/laat gpu=6 callbacks=no_wandb trainer.print_metrics=true`
   
#### Evaluation
If you just want to evaluate the models using the provided model_checkpoints you need to do set `trainer.epochs=0` and provide the path to the models checkpoint `load_model=path/to/model_checkpoint`. Make sure you the correct model-checkpoint with the correct configs.

Example:
Evaluate PLM-ICD on MIMIC-IV ICD-10 on GPU 1: `python main.py experiment=mimiciv_icd10/plm_icd gpu=1 load_model=path/to/model_checkpoints/mimiciv_icd10/plm_icd epochs=0`

