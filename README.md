## Problem Overview
Hate speech is detection often has to deal performing classification on imbalance dataset. The generative models such as GAN and VAE have show promising results in generating synthetic datasets to help supplement datasets to overcome the imbalance data problem. In this study, we want to explore how the different generative models can help generate posts to overcome the imbalance dataset problem in hate speech detection.

## Experiments for Baseline Models

Data Distribution  
- For combined dataset, hate from Hate Lingo, Normal from SemEval, Harassment and FOUNTA

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (3,435) non-hate (9,767)                     |
| DT      | Multi  | hate (1,430) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (3,907) abusive (19,232) normal (1,430)      |
| COMBINED| Binary | hate (1,124) non-hate (823,915)                   |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary | 78.19|79.33|76.34|62.2       |45.8      |46.6     |
| DT      | Multi  | 88.98|90.02|89.02|54.6       |23.8      |32.0     |
| FOUNTA  | Multi  | 90.72|91.65|90.91|26.9       |26.2      |35.0     |  
| COMBINE | Binary | 98.94|98.96|98.94|92.6       |83.2      |88.0     |

## Experiments for HateGAN Method 1: Generate Hate Speech Only
In this method, we will use GAN model to generate tweets that contain hate speech.

Data Distribution after adding the hate speech: 1099

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (4,534) non-hate (9,767)                     |
| DT      | Multi  | hate (2,529) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (5,006) abusive (19,232) normal (1,430)      |
| COMBINED| Binary | hate (2,223) non-hate (823,915)                   |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary |79.96 |80.45|77.42|67.8       |46.4      |48.2     |
| DT      | Multi  |89.29 |90.15|89.45|53.0       |29.6      |37.2     |
| FOUNTA  | Multi  |90.59 |91.58|90.86|51.2       |26.8      |35.0     |  
| COMBINE | Binary |98.41 |98.33|98.36|97.8       |95.4      |96.6     |

## Experimetns for HateGAN Method 2: Generate Posts then Extract Hate Speech
In this method, we will generate unlabeled tweets and use a pre-trained classifier to extract the hate tweets from the generate tweets.

Data Distribution after adding the hate speech: 1072

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (4,507) non-hate (9,767)                     |
| DT      | Multi  | hate (2,502) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (4,979) abusive (19,232) normal (1,430)      |  
| COMBINED| Binary | hate (2,196) non-hate (823,915)                   |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary |80.06  |80.67  |77.81  |69.0 |45.6|49.4      |
| DT      | Multi  |89.20  |90.02|89.17 |55.2  |27.8 |34.6    |
| FOUNTA  | Multi  |90.52  |91.56|90.75|53.0|25.2   |34.0   |  
| COMBINE | Binary |99.10 |99.11|99.10|97.8       |95.4      |96.6     |
