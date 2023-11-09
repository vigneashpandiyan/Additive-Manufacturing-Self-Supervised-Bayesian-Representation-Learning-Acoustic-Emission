# Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning
Self-Supervised Bayesian Representation Learning of Acoustic Emissions from Laser Powder Bed Fusion Process for In-situ Monitoring
![Graphical abstract](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning/assets/39007209/638e63cf-8004-486a-9f3e-cfbcce913a04)
# Journal link
https://doi.org/10.1016/j.jmapro.2022.07.033

![LPBF](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/aa6fa98d-a0c8-4424-8fbf-aae661a5bdbd)

# Overview

Although most LPBF monitoring strategies based on process zone information in the literature for LPBF process are trained in supervised, unsupervised and semi-supervised manners, the authors of this work take a first step towards creating a framework for monitoring part quality in terms of build density using a self-supervisedly trained Bayesian Neural Network (BNN). The motivation for this approach stems from the challenges of labeling datasets with discrete process dynamics and semantic complexities. Self-supervised models offer a fully unsupervised training opportunity, which reduces the time and cost associated with algorithm setup. Furthermore, self-supervised learning can facilitate transfer learning, where a pre-trained model is fine-tuned for a specific task, thus minimizing the amount of labeled data required and enhancing efficiency, which is also demonstrated in this work. 

![Picture1](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/f87c5814-c174-4098-bc22-3526563cd62c)

# Bayesian Representation Learning

Deep learning models are created to understand the relationships between data samples in order to make predictions about the objectives for which they were trained. Thanks to recent improvements in self-supervised representation learning, models can now be trained on less annotated data samples. The goal of self-supervised learning is to identify the most informative characteristics of unlabelled data by creating a supervisory signal, which leads to the learning of generalizable representations. Self-supervised learning has been successful in various computer vision tasks. The self-supervised representation introduced in this study draws inspiration from prior works [64-67] and offers a powerful method for decoding inter and intra-temporal relationships. The methodology proposed aims to extract time series representations from unlabeled data through inter-sample and intra-temporal relation reasoning. This is accomplished by utilizing a shared representation learning encoder backbone (f_( θ)) based on Bayesian Neural Network (BNN), as depicted in Figure below. 

![image](https://github.com/vigneashpandiyan/Additive-Manufacturing-Transfer-Learning/assets/39007209/0ceb2fa9-8cae-4abf-a4a3-7fd3a85050d8)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning
cd Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning
python Main.py
```

# Citation
```
@article{PANDIYAN2023112458,
title = {Self-Supervised Bayesian representation learning of acoustic emissions from laser powder bed Fusion process for in-situ monitoring},
journal = {Materials & Design},
volume = {235},
pages = {112458},
year = {2023},
issn = {0264-1275},
doi = {https://doi.org/10.1016/j.matdes.2023.112458},
url = {https://www.sciencedirect.com/science/article/pii/S0264127523008730},
publisher={Elsevier}
}
