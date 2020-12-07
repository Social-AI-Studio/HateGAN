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

## Getting Started  

- Pretrain the toxic comment detection model, use code in Toxic Model.ipynb

- Train the generative model, use code in HateGAN:  

    ``` python main.py ``` 

- Evaluate the generated tweets, use code in Test-GAN-gen:  

    ``` python main.py ``` 

We have also reference the SeqGAN implementation: <https://github.com/X-czh/SeqGAN-PyTorch>.

## Acknowledge  
To cite:
```
@inproceedings{cao2020hategan,
  title={HateGAN: Adversarial Generative-Based Data Augmentation for Hate Speech Detection},
  author={Cao, Rui and Lee, Roy Ka-Wei},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6327--6338},
  year={2020}
}
```


