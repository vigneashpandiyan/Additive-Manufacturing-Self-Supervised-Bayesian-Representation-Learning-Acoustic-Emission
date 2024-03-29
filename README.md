# Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning
Self-Supervised Bayesian Representation Learning of Acoustic Emissions from Laser Powder Bed Fusion Process for In-situ Monitoring

![Abstract](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning-Acoustic-Emission/assets/39007209/c52164a8-b0c2-4fc7-bbb9-efc58cbd7d28)
# Journal link
https://doi.org/10.1016/j.matdes.2023.112458

![LPBF](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/aa6fa98d-a0c8-4424-8fbf-aae661a5bdbd)

# Overview

Although most LPBF monitoring strategies based on process zone information in the literature for LPBF process are trained in supervised, unsupervised and semi-supervised manners, the authors of this work take a first step towards creating a framework for monitoring part quality in terms of build density using a self-supervisedly trained Bayesian Neural Network (BNN). This study presents a self-supervised BNN framework using air-borne Acoustic Emission (AE) to identify different Laser Powder Bed Fusion (LPBF) process regimes such as Lack of Fusion, conduction mode, and keyhole without ground-truth information. The proposed framework addresses the challenge of labelling datasets with semantic complexities into discrete process dynamics. This novel AE-based in-situ monitoring approach provides a promising alternative to quantify part density in LPBF process. The study demonstrates the effectiveness of a Bayesian encoder backbone for learning the manifold representations of LPBF regimes, which were visually separable in a lower-dimensional representation using t-distributed stochastic neighbour embedding. The generalized representations learned by the Bayesian backbone allowed traditional classifiers trained on smaller datasets to exhibit high classification accuracy. The feature map computed using pre-trained Bayesian encoder on other datasets was also effective in anomaly detection, achieving 92% accuracy with one-class Support Vector Machine. Additionally, the representation learned by the BNN facilitates transfer learning, where it can be fine-tuned for classification tasks on different process maps, which is also demonstrated in this work. Our proposed framework improves the generalization and robustness of the LPBF monitoring, particularly in the face of varying data distribution across multiple process parameter spaces.

![Picture1](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/f87c5814-c174-4098-bc22-3526563cd62c)

# Bayesian Representation Learning

Deep learning models are created to understand the relationships between data samples in order to make predictions about the objectives for which they were trained. Thanks to recent improvements in self-supervised representation learning, models can now be trained on less annotated data samples. The goal of self-supervised learning is to identify the most informative characteristics of unlabelled data by creating a supervisory signal, which leads to the learning of generalizable representations. Self-supervised learning has been successful in various computer vision tasks. The self-supervised representation introduced in this study draws inspiration from prior works and offers a powerful method for decoding inter and intra-temporal relationships. The methodology proposed aims to extract time series representations from unlabeled data through inter-sample and intra-temporal relation reasoning. This is accomplished by utilizing a shared representation learning encoder backbone (f_( θ)) based on Bayesian Neural Network (BNN), as depicted in Figure below. 

![image](https://github.com/vigneashpandiyan/Additive-Manufacturing-Transfer-Learning/assets/39007209/0ceb2fa9-8cae-4abf-a4a3-7fd3a85050d8)

# Highlights

* The study fills a research gap regarding the robustness of LPBF monitoring when faced with varying data distribution across multiple process parameter spaces.
* Addressing the challenge of labeling datasets with semantic complexities into discrete process dynamics, the study proposes an ML strategy that utilizes air-borne Acoustic Emission (AE) from the process zone.
* The study proposes a self-supervised representation learning framework using a Bayesian Neural Network to identify LPBF process dynamics without ground-truth information.
* The framework's effectiveness is highlighted in classification, anomaly detection, and transfer learning, even in the presence of offset in the AE data associated with different LPBF parameters.
* The study demonstrated the enhanced generalizability of the ML model by showcasing the prediction accuracy of the proposed self-supervised learning methodology in the newer environment.

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
