## Problem Overview
Hate speech is detection often has to deal performing classification on imbalance dataset. The generative models such as GAN and VAE have show promising results in generating synthetic datasets to help supplement datasets to overcome the imbalance data problem. In this study, we want to explore how the different generative models can help generate posts to overcome the imbalance dataset problem in hate speech detection.

## Experiments for Baseline Models

Data Distribution

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (3,435) non-hate (9,767)                     |
| DT      | Multi  | hate (1,430) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (3,907) abusive (19,232) normal (1,430)      |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary | 78.19|79.33|76.34|62.2       |45.8      |46.6     |
| DT      | Multi  | 88.98|90.02|89.02|54.6       |23.8      |32.0     |
| FOUNTA  | Multi  | 90.72|91.65|90.91|26.9       |26.2      |35.0      |

## Experiments for HateGAN Method 1: Generate Hate Speech Only
In this method, we will use GAN model to generate tweets that contain hate speech.

Data Distribution after adding the hate speech: 1099

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | ???                                               |
| DT      | Multi  | ???                                               |
| FOUNTA  | Multi  | ???                                               |

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary | ???  |???  |???  |???        |???       |???      |
| DT      | Multi  | ???  |???  |???  |???        |???       |???      |
| FOUNTA  | Multi  | ???  |???  |???  |???        |???       |???      |

## Experimetns for HateGAN Method 2: Generate Posts then Extract Hate Speech
In this method, we will generate unlabeled tweets and use a pre-trained classifier to extract the hate tweets from the generate tweets.

Data Distribution after adding the hate speech: 1072

| Dataset | Type   | Label (Count)                                     |
| :-----: | :----: | :-----------------------------------------------: | 
| WZ      | Binary | hate (4,507) non-hate (9,767)                     |
| DT      | Multi  | hate (2,502) offensive (19,190) beither (4,163)   |
| FOUNTA  | Multi  | hate (4,979) abusive (19,232) normal (1,430)      |  

Results

| Dataset | Type   | Prec | Rec | F1  | Prec-Hate | Rec-Hate | F1-Hate |
| :-----: | :----: | :--: | :-: | :-: | :-------: | :------: | :-----: |
| WZ      | Binary |80.06  |80.67  |77.81  |69.0 |45.6|49.4      |
| DT      | Multi  |89.20  |90.02|89.17 |55.2  |27.8 |34.6    |
| FOUNTA  | Multi  |90.52  |91.56|90.75|53.0|25.2   |34.0   |
