# DarkSAM

The implementation of our NeurIPS 2024 paper "DarkSAM: Fooling Segment Anything Model to Segment Nothing".


![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-red.svg?style=plastic)


## Abstract

Segment Anything Model (SAM) has recently gained much attention for its outstanding generalization to unseen data and tasks. Despite its promising prospect, the vulnerabilities of SAM, especially to universal adversarial perturbation (UAP) have not been thoroughly investigated yet. In this paper, we propose DarkSAM, the first prompt-free universal attack framework against SAM, including a semantic decoupling-based spatial attack and a texture distortion-based frequency attack. We first divide the output of SAM into foreground and background. Then, we design a shadow target strategy to obtain the semantic blueprint of the image as the attack target. DarkSAM is dedicated to fooling SAM by extracting and destroying crucial object features from images in both spatial and frequency domains. In the spatial domain, we disrupt the semantics of both the foreground and background in the image to confuse SAM. In the frequency domain, we further enhance the attack effectiveness by distorting the high-frequency components (i.e., texture information) of the image. Consequently, with a single UAP, DarkSAM renders SAM incapable of segmenting objects across diverse images with varying prompts. Experimental results on four datasets for SAM and its two variant models demonstrate the powerful attack capability and transferability of DarkSAM.



<img src="pipeline.png"/>

Our code is currently in preparation and will be released shortly!


## BibTeX 
If you find DarkSAM both interesting and helpful, please consider citing us in your research or publications:
```bibtex
@inproceedings{zhou2024dark,
  title={DarkSAM: Fooling Segment Anything Model to Segment Nothing},
  author={Zhou, Ziqi and Song, Yufei and Li, Minghui and Hu, Shengshan and Wang, Xianlong and Zhang, Leo Yu and Yao, Dezhong and Jin, Hai},
  booktitle={Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS'24)},
  year={2024}
}

```



