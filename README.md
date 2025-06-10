# A Hierarchical Architecture for Neural Materials

[![Paper](https://img.shields.io/badge/Paper-CGF%202024-blue)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15116?af=R)
[![arXiv](https://img.shields.io/badge/arXiv-2307.10135-b31b1b.svg)](https://arxiv.org/abs/2307.10135)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/yley123/A-Hierarchical-Architecture-for-Neural-Materials)

[Bowen Xue](mailto:bowen.xue@postgrad.manchester.ac.uk)<sup>1</sup>, 
[Shuang Zhao](https://shuangz.com)<sup>2</sup>, 
[Henrik Wann Jensen](https://henrikwann.com)<sup>3</sup>, 
[Zahra Montazeri](mailto:zahra.montazeri@manchester.ac.uk)<sup>1</sup>

<sup>1</sup>University of Manchester, <sup>2</sup>UC Irvine, <sup>3</sup>Luxion Inc

This repository contains the official implementation of our Computer Graphics Forum 2024 paper "A Hierarchical Architecture for Neural Materials".

## 📰 News
- **[June 2024]** Paper accepted to Computer Graphics Forum!

## 🎯 Abstract

Neural reflectance models are capable of reproducing the spatially-varying appearance of many real-world materials at different scales. Unfortunately, existing techniques such as NeuMIP have difficulties handling materials with strong shadowing effects or detailed specular highlights. In this paper, we introduce a neural appearance model that offers a new level of accuracy. Central to our model is an inception-based core network structure that captures material appearances at multiple scales using parallel-operating kernels and ensures multi-stage features through specialized convolution layers. Furthermore, we encode the inputs into frequency space, introduce a gradient-based loss, and employ it adaptive to the progress of the learning phase.

## ✨ Key Features

- 🏗️ **Hierarchical Architecture**: Inception-based decoder that captures multi-scale features
- 🌊 **Frequency Encoding**: Fourier features for high-frequency detail preservation  
- 📊 **Enhanced Loss Functions**: Gradient loss and output remapping for better shadow/highlight accuracy
- 🎯 **88% Error Reduction**: Compared to NeuMIP baseline with only 25% additional compute
- 🔍 **Multi-Resolution**: Accurate rendering at all levels of detail

## 🚀 Getting Started

### Quick Start
```bash
# Train a neural material model
python neural_rendering.py --dataset /your/dataset/path
```

## 📖 Citation

If you find our work useful, please cite:
```bibtex
@article{xue2024hierarchical,
  author = {Xue, Bowen and Zhao, Shuang and Jensen, Henrik Wann and Montazeri, Zahra},
  title = {A Hierarchical Architecture for Neural Materials},
  journal = {Computer Graphics Forum},
  volume = {43},
  number = {6},
  pages = {e15116},
  year = {2024},
  doi = {https://doi.org/10.1111/cgf.15116}
}
```

## 📮 Contact

For questions and feedback:
- Bowen Xue: bowen.xue@postgrad.manchester.ac.uk
- Project page: [https://github.com/yley123/A-Hierarchical-Architecture-for-Neural-Materials](https://github.com/yley123/A-Hierarchical-Architecture-for-Neural-Materials)
