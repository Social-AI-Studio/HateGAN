## Problem Overview
Most of existing methods for hate speech detection adopt a supervised aproach that heavily depends on labeled datasets for training. The imbalance of datasets may be detrimental to performance of methods. We propose HateGAN, a deep generative reinforcement learning model to address the challenge of imbalance class by augementing datasets with hateful tweets.  

This repository contains code for the implementation of HateGAN.

## Dependencies:  

- Python 3.5
- Pytorch 1.0.0

## Data for Training HateGAN

| Dataset | Label (Count)                                     |
| :-----: | :-----------------------------------------------: | 
| WZ      | hate (3,435) non-hate (9,767)                     |
| DT      | hate (1,430) offensive (19,190) beither (4,163)   |
| FOUNTA  | hate (3,907) abusive (19,232) spam (13,840) normal (53,011)      |  
| HateLingo      | hate (11,763)                     |

## Data for Hate Speech Detection

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (3,435) non-hate (9,767)                     |
| DT      | Multi  | hate (1,430) offensive (19,190) beither (4,163)   |  

## Experiments for Baseline Models

Data Distribution  
- For combined dataset, hate from Hate Lingo, Normal from SemEval, Harassment and FOUNTA

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (3,435) non-hate (9,767)                     |
| DT      | Multi  | hate (1,430) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (3,907) abusive (19,232) normal (1,430)      |
| COMBINED| Multi  | hate (1,995) normal (15,075) offensive (19,389)   |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary | 78.19|79.33|76.34|62.2       |45.8      |46.6     |
| DT      | Multi  | 88.98|90.02|89.02|54.6       |23.8      |32.0     |
| FOUNTA  | Multi  | 90.72|91.65|90.91|26.9       |26.2      |35.0     |  
| COMBINE | Multi  | 90.78|91.08|90.85|62.8       |49.0      |88.0     |

## Getting Started  

- Pretrain the toxic comment detection model, use code in Toxic Model.ipynb

- Train the generative model, use code in HateGAN:  

    ``` python main.py ``` 

- Evaluate the generated tweets, use code in Test-GAN-gen:  

    ``` python main.py ``` 


## Reference  
Referred paper:
```
@inproceedings{DBLP:conf/aaai/YuZWY17,
  author    = {Lantao Yu and Weinan Zhang and Jun Wang and Yong Yu},
  title     = {SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient},
  booktitle = {Conference on Artificial Intelligence,},
  pages     = {2852--2858},
  publisher = {{AAAI} Press},
  year      = {2017}
}
```   

Referred Code:
<https://github.com/X-czh/SeqGAN-PyTorch>


## Acknowledge  
To cite:
```
@inproceedings{cao2020hategan,
  title={HateGAN: Adversarial Generative-Based Data Augmentation for Hate Speech Detection},
  author={Cao, Rui and Lee, Roy Ka-Wei},
  booktitle={The 28th International Conference on Computational Linguistics},
  year={2020}
