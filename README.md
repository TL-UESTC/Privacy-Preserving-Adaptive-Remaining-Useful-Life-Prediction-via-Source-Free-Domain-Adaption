# Privacy-Preserving-Adaptive-Remaining-Useful-Life-Prediction-via-Source-Free-Domain-Adaption
This is the official project repository for [Privacy-Preserving Adaptive Remaining Useful Life Prediction via Source-Free Domain Adaption](https://ieeexplore.ieee.org/abstract/document/10239252)) by Kangkai Wu , Jingjing Li, Lichao Meng, Fengling Li, Heng Tao Shen (TIM 2023).

Abstract: Unsupervised domain adaptation (UDA) strives to transfer the learned knowledge to differently distributed datasets by utilizing both source and target data. Recently, an increasing number of UDA methods have been proposed for domain adaptive remaining useful lifetime (RUL) prediction. However, many industries value their privacy protection a lot. The confidentiality of degradation data in certain fields, such as aircraft engines or bearings, makes the source data inaccessible. To cope with this challenge, our work proposes a source-free domain adaption method to implement cross-domain RUL prediction. Especially, an adversarial architecture with one feature encoder and two RUL predictors is proposed. We first maximize the prediction discrepancy between the predictors to detect target samples that are far from the support of the source. Then the feature encoder is trained to minimize the discrepancy, which can generate features near the support. Besides, a weight regularization is utilized to replace the supervised training on the source domain. We evaluate our proposed approach on the commonly used C-MAPSS and FEMTO-ST datasets. Extensive experiment results demonstrate that our approach can significantly improve the prediction reliability on the target domain.

![Step](https://s2.loli.net/2023/09/21/lERueVvbx3Jotc4.png)

## Usage

* Conda Enviroment

    `conda env create -f environment.yaml`

* For Pretraining

    `python SFDA-RUL\trainer\pretrain_phm.py`

* For Cross-domain Training

    `python SFDA-RUL\trainer\main.py`

The processed data can be downloaded from this [LINK](https://drive.google.com/drive/folders/12vxOBouxJlrdfDTa0jCCTb5MQ6ccZ-2O?usp=sharing).

## Results
![Result1](https://s2.loli.net/2023/09/21/tKhdiPwUj8BZ7q6.jpg)
![Result2](https://s2.loli.net/2023/09/21/NoRVubG1ImgW2qH.jpg)

## Citation
```
@article{wu2023privacy,
  title={Privacy-Preserving Adaptive Remaining Useful Life Prediction via Source Free Domain Adaption},
  author={Wu, Kangkai and Li, Jingjing and Meng, Lichao and Li, Fengling and Shen, Heng Tao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```
