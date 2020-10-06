## Problem Overview
Hate speech is detection often has to deal performing classification on imbalance dataset. The generative models such as GAN and VAE have show promising results in generating synthetic datasets to help supplement datasets to overcome the imbalance data problem. In this study, we want to explore how the different generative models can help generate posts to overcome the imbalance dataset problem in hate speech detection.


Most of existing methods for hate speech detection adopt a supervised aproach that heavily depends on labeled datasets for training. The imbalance of datasets may be detrimental to performance of methods. 
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

## Experiments for HateGAN Method 1: Generate Hate Speech Only
In this method, we will use GAN model to generate tweets that contain hate speech.

Data Distribution after adding the hate speech: 1099

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (4,534) non-hate (9,767)                     |
| DT      | Multi  | hate (2,529) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (5,006) abusive (19,232) normal (1,430)      |
| COMBINED| Multi  | hate (3,094) normal (15,075) offensive (19,389)   |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary |79.96 |80.45|77.42|67.8       |46.4      |48.2     |
| DT      | Multi  |89.29 |90.15|89.45|53.0       |29.6      |37.2     |
| FOUNTA  | Multi  |90.59 |91.58|90.86|51.2       |26.8      |35.0     |  
| COMBINE | Multi  |90.59 |90.78|90.64|56.2       |48.6      |52.0     |

## Reference  
Referred paper:
'''
@inproceedings{DBLP:conf/aaai/YuZWY17,
  author    = {Lantao Yu and
               Weinan Zhang and
               Jun Wang and
               Yong Yu},
  title     = {SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient},
  booktitle = {Conference on Artificial Intelligence,},
  pages     = {2852--2858},
  publisher = {{AAAI} Press},
  year      = {2017}
}
## Acknowledge
To cite:
```
@inproceedings{cao2020hategan,
  title={HateGAN: Adversarial Generative-Based Data Augmentation for Hate Speech Detection},
  author={Cao, Rui and Lee, Roy Ka-Wei},
  booktitle={The 28th International Conference on Computational Linguistics},
  pages={11--20},
  year={2020}
| COMBINE | Multi  |90.51 |90.74|90.54|57.6       |47.0      |51.4     |
