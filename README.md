# ⚕️Automatic Medical Coding

## Introduction 
Automatic medical coding is the task of automatically assigning diagnosis and procedure codes based on discharge summaries from electronic health records. This repository contains code for easily implementing new automatic medical coding machine learning models Multiple machine learning models for automatic medical coding on [MIMIC-III](https://www.nature.com/articles/sdata201635) have been implemented in this codebase. The following models have been implemented:

| Model | Paper | Original Code |
| ----- | ----- | ------------- |
| CNN   |[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
| Bi-GRU|[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
|CAML   |[Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) | [link](https://github.com/jamesmullenbach/caml-mimic) | 
| MultiResCNN | [ICD Coding from Clinical Text Using Multi-Filter Residual Convolutional Neural Network](https://arxiv.org/pdf/1912.00862.pdf) | [link](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network) |
| LAAT | [A Label Attention Model for ICD Coding from Clinical Text](https://arxiv.org/abs/2007.06351) | [link](https://github.com/aehrc/LAAT) |
| PLM-ICD | [PLM-ICD: Automatic ICD Coding with Pretrained Language Models](https://aclanthology.org/2022.clinicalnlp-1.2/) | [link](https://github.com/MiuLab/PLM-ICD) |


## How to reproduce results
1. Download the MIMIC-III data into your preferred location `your/preferred/raw/data/location`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours.  - [link to data access](https://physionet.org/content/mimiciii/1.4/)
2. Create a conda environement `conda create -n coding python=3.9`
3. Install the packages `pip install . -e`
4. Open the file `automatic_medical_coding/settings.py`
5. Change the variable `DOWNLOAD_DIRECTORY` to the path of your downloaded data `your/preferred/raw/data/location`
6. Change the variable `DATA_DIRECTORY_MULLENBACH` and `DATA_DIRECTORY_CLEAN` to where you want to store the prepared data.
7. If you want to use the MIMIC-III full and MIMIC-III 50 from the [Explainable Prediction of Medical Codes from Clinical Text](https://aclanthology.org/N18-1100/) you need to run `python prepare_data/prepare_mimiciii_mullenbach.py`
8. If you want to use MIMIC-III clean from our paper you need to run `python prepare_data/prepare_mimiciii.py`
9. Setup Weights and Biases.
9. You can now run any experiment found in `configs/experiment`. Here are some examples:
    * Train PLM-ICD on MIMIC-III clean on GPU 0: `python main.py experiment=mimiciii_clean/plm_icd gpu=0`
    * Train CAML on MIMIC-III full on GPU 6: `python main.py experiment=mimiciii_full/caml gpu=6`



## Hydra

## Weights and Biases
