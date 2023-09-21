# Privacy-Preserving-Adaptive-Remaining-Useful-Life-Prediction-via-Source-Free-Domain-Adaption
Kangkai Wu , Jingjing Li, Lichao Meng, Fengling Li, Heng Tao Shen

Abstract: Unsupervised domain adaptation (UDA) strives to transfer the learned knowledge to differently distributed datasets by utilizing both source and target data. Recently, an increasing number of UDA methods have been proposed for domain adaptive remaining useful lifetime (RUL) prediction. However, many industries value their privacy protection a lot. The confidentiality of degradation data in certain fields, such as aircraft engines or bearings, makes the source data inaccessible. To cope with this challenge, our work proposes a source-free domain adaption method to implement cross-domain RUL prediction. Especially, an adversarial architecture with one feature encoder and two RUL predictors is proposed. We first maximize the prediction discrepancy between the predictors to detect target samples that are far from the support of the source. Then the feature encoder is trained to minimize the discrepancy, which can generate features near the support. Besides, a weight regularization is utilized to replace the supervised training on the source domain. We evaluate our proposed approach on the commonly used C-MAPSS and FEMTO-ST datasets. Extensive experiment results demonstrate that our approach can significantly improve the prediction reliability on the target domain.

![Idea.jpg](file:///D:/SFDA-RUL/Snipaste_2023-09-21_19-52-56.jpg)

## Usage

* Conda Enviroment

    `conda env create -f environment.yaml`

* For Pretraining

    `python SFDA-RUL\trainer\pretrain_phm.py`

* For Cross-domain Training

    `python SFDA-RUL\trainer\main.py`

The processed data can be downloaded from this [LINK](https://drive.google.com/drive/folders/12vxOBouxJlrdfDTa0jCCTb5MQ6ccZ-2O?usp=sharing).

## Results

