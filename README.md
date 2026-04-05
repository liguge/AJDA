# AJDA: Asynchronous Joint Distribution Alignment for Cross-Domain Fault Diagnosis

**异步联合分布适配 | 机械故障诊断迁移学习**

Official implementation of AJDA (Asynchronous Joint Distribution Alignment) for cross-condition/cross-bearing fault diagnosis in rotating machinery (planetary gearboxes, bearings).

[[GitHub Repository](https://github.com/liguge/AJDA)] | Paper: [IEEE/ASME TRANSACTIONS ON MECHATRONICS](https://doi.org/10.1109/TMECH.2026.3669233)

## 📖Article Interpretation

Brief Version:

- Wechat：

## 🔍Overview

Distribution alignment mechanism serves as a fundamental basis for a cross-domain fault diagnosis, directly affecting diagnostic accuracy. A typical joint distribution alignment which enables a fine-grained class-level subdomain alignment, has achieved notable success in some cases. However, this mechanism performs a synchronous class-wise alignment across all categories between the source and the target domains, without considering the confidence of pseudolabels. This approach may lead to the collapse of the entire knowledge transfer framework due to an unreliable alignment. By contrast, an asynchronous joint distribution alignment (AJDA) mechanism is developed to enhance the model stability. In AJDA, the asynchronous class-level alignment strategy is first proposed to adaptively select high-confidence class-conditional distributions from the training process. Furthermore, a new highorder significant discrepancy representation metric based on the mechanical vibration characteristics is constructed to improve the sensitivity in measuring cross-domain distribution shifts. Finally, the proposed AJDA mechanism is validated through two scenarios, including planetary gearbox cross-condition and public cross-bearing transfer diagnosis cases, achieving average diagnostic accuracies of 98.81% and 80.23%, respectively. AJDA yields an overall performance improvement of more than 20% compared with other well-known methods, confirming its effectiveness and advantage.

## ✨ Key Features

- **✅ Asynchronous Class-Wise Alignment**: Adaptively selects high-confidence class-conditional distributions for alignment, mitigating the risk of framework collapse caused by unreliable pseudolabels.
- **✅ High-Order Significant Discrepancy (HoSDR)**: Outperforms traditional first/second-order distribution metrics (MMD, CORAL) in capturing non-Gaussian vibration signal features, especially under noise interference.
- **✅ Plug-and-Play Module**: The AJDA adaptation module can be easily embedded into various CNN backbones for different fault diagnosis scenarios.

## 📄 Paper

- **Title**: Asynchronous Joint Distribution Alignment: A New Domain Confusion Mechanism for Fault Transfer Diagnosis

- **Authors**: Quan Qian, Yang Yu, Yi Qin, Jiusi Zhang, Yuhua Cheng, Kai Chen

- **DOI**: [10.1109/TMECH.2026.3669233](https://doi.org/10.1109/TMECH.2026.3669233)

- **GitHub**: [https://github.com/liguge/AJDA](https://github.com/liguge/AJDA)

## 🔧 Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/liguge/AJDA.git
cd AJDA
```

### Prerequisites

The code is tested on Python 3.7+, PyTorch 1.8.0+. Required dependencies:

```plain text
python >= 3.7
torch >= 1.8.0
numpy >= 1.19.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
pandas >= 1.1.0
tqdm >= 4.60.0
```

Note: If you encounter version conflicts, please adjust the package versions according to your local environment.

## 📊 Datasets

The AJDA model is validated on two representative scenarios, covering cross-load and cross-bearing transfer diagnosis tasks, as described in the paper:

- **Planetary Gearbox Dataset (Private)**: A planetary gearbox fault dataset is collected using a private testbed (see Fig. 6). This testbed consists of a three-phase motor, a two-stage planetary gearbox, a parallel-shaft gearbox, and a magnetic powder brake. An acceleration sensor is mounted on the housing of the two-stage planetary gearbox to acquire vibration signals under various health conditions, with a sampling frequency of 5120 Hz. Faults are artificially introduced on the sun gear, covering five health states: normal condition; surface wear; root crack; chipped tooth; and missing tooth (MT). The input speed of the planetary gearbox is set to 1500 r/min. Different load conditions, including 0, 1.4, 2.8, and 25.2 N·m, are simulated by adjusting the current of the magnetic powder brake. In each condition, the sampling duration is 200 s.

- **Public Bearing Datasets**: 
  - CWRU (Case Western Reserve University)

    - JNU (Jiangnan University)

    - SEU (Southeast University)

- **Custom Datasets**: Supports user-defined industrial bearing/gearbox fault datasets (see **Data Preparation** for details).

You can download the public datasets from their official sources or contact the paper authors for dataset access.

## 📁 Data Preparation

For real mechanical fault vibration signal data, please organize the data according to the format required by the data loader in `data/dataset`

## 📂 Code Structure

The project structure is organized as follows for clarity and maintainability:

```plain text
AJDA/
├── AJDA.ipynb                  # Main Jupyter notebook with complete implementation
├── README.md                   # Project documentation
├── train_advanced.py           # Advanced training script
├── datasets/                   # Data loading and preprocessing module
│   ├── __init__.py
│   ├── JNU.py                 # JNU bearing dataset loader
│   ├── JNUFFT.py              # JNU dataset with FFT preprocessing
│   ├── PU.py                  # Paderborn University dataset loader
│   ├── SequenceDatasets.py    # Generic sequence dataset wrapper
│   └── sequence_aug.py        # Sequence data augmentation
├── loss/                       # Domain adaptation loss functions
│   ├── __init__.py
│   ├── AJDA.py                # Asynchronous Joint Distribution Alignment loss (proposed method)
│   ├── CORAL.py               # CORrelation ALignment loss
│   ├── DAN.py                 # Deep Adaptation Network loss
│   └── JAN.py                 # Joint Adaptation Network loss
├── models/                     # Neural network architectures
│   ├── __init__.py
│   ├── CNN.py                 # 1D Convolutional Neural Network
│   └── resnet18_1d.py         # 1D ResNet-18 backbone
└── utils/                      # Utility functions
    ├── __init__.py
    ├── entropy_CDA.py         # Entropy-based conditional domain alignment
    ├── logger.py              # Logging utilities
    ├── train_utils_base.py    # Base training utilities
    └── train_utils_combines.py # Combined training utilities for domain adaptation
```

## 📈 Main Results

- TABLE I EXPERIMENTAL DIAGNOSTIC RESULTS ON THE PLANETARY GEARBOX DATASETS

| Methods  | 0→1.4 N·m    | 1.4→0 N·m      | 0→2.8 N·m      | 2.8→0 N·m    | 0→25.2 N·m     | 25.2→0 N·m     | Average   |
| -------- | ------------ | -------------- | -------------- | ------------ | -------------- | -------------- | --------- |
| DDC      | 95.43±1.43   | 94.84±1.44     | 81.27±1.42     | 80.45±2.04   | 35.75±2.78     | 38.81±3.08     | 71.09     |
| DCORAL   | 95.01±0.76   | 97.15±1.01     | 86.22±0.87     | 86.65±0.94   | 50.43±1.45     | 47.28±2.03     | 77.12     |
| DANN     | 93.11±0.65   | 96.40±0.88     | 84.23±0.84     | 82.19±2.32   | 42.60±1.84     | 36.17±0.90     | 72.45     |
| MCD      | 92.84±1.05   | 95.37±0.87     | 84.20±1.64     | 80.54±0.74   | 32.15±2.51     | 40.52±1.12     | 70.94     |
| DTN-JDA  | 95.52±1.87   | 95.69±1.23     | 83.35±1.11     | 83.78±1.46   | 35.25±1.83     | 38.88±3.20     | 72.08     |
| AICDA    | 93.54±1.54   | 95.23±0.89     | 84.66±1.20     | 90.57±0.25   | 56.59±2.99     | 46.96±1.65     | 77.93     |
| DJDA     | 98.78±0.59   | 96.18±1.05     | 80.77±1.51     | 87.02±1.03   | 72.15±1.44     | 75.51±2.61     | 85.07     |
| **AJDA** | **100±0.00** | **99.78±0.04** | **99.25±0.14** | **100±0.00** | **96.58±1.36** | **97.23±1.87** | **98.81** |

- TABLE III EXPERIMENTAL DIAGNOSTIC RESULTS ON THE THREE PUBLIC BEARING DATASETS

  | Methods  | A→B            | B→A            | A→C            | C→A            | B→C            | C→B            | Average   |
  | -------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------- |
  | DDC      | 33.92±2.50     | 53.08±1.55     | 62.45±3.04     | 55.09±2.46     | 21.24±2.20     | 22.11±1.94     | 41.31     |
  | DCORAL   | 47.97±1.32     | 58.75±1.55     | 39.14±1.76     | 48.78±2.01     | 18.14±1.12     | 13.53±0.96     | 37.72     |
  | DANN     | 49.75±2.08     | 39.89±3.24     | 41.76±4.76     | 41.48±4.76     | 39.12±6.37     | 49.92±4.80     | 43.65     |
  | MCD      | 28.04±1.29     | 42.49±4.07     | 24.02±4.22     | 24.01±2.65     | 30.77±3.31     | 31.75±3.61     | 30.18     |
  | DTN-JDA  | 32.56±1.97     | 41.63±3.90     | 31.02±2.54     | 39.87±3.43     | 26.98±4.21     | 23.21±2.75     | 32.55     |
  | AICDA    | 43.12±2.14     | 54.25±2.57     | 58.08±1.82     | 50.10±2.13     | 33.00±3.77     | 33.21±3.03     | 45.29     |
  | DJDA     | 46.35±1.17     | 38.13±3.39     | 39.87±2.45     | 41.28±3.11     | 36.98±1.36     | 42.69±1.99     | 40.88     |
  | **AJDA** | **76.98±2.30** | **72.84±2.36** | **87.56±3.10** | **90.16±3.56** | **73.85±2.78** | **80.00±3.64** | **80.23** |

## 💡 Key Innovations

1. **Theoretical Analysis of JDA**: Rigorously proves that the loss function of traditional JDA is a sufficient but not necessary condition for true joint distribution alignment, laying a theoretical foundation for AJDA.
2. **Asynchronous Class-Wise Alignment Strategy**: Uses Distribution Overlap Area (DOA) to evaluate pseudolabel confidence, only aligning high-confidence class-conditional distributions (DOA > threshold and maximum DOA) to avoid negative transfer.
3. **High-Order Significant Discrepancy Representation (HoSDR)**: Combines mean square statistics and tensor product to capture nonstationary and nonlinear vibration signal features, improving sensitivity to distribution shifts under noise.

## 📝 Citation

If you use this code or the AJDA method in your research, please cite the original paper:

```bibtex
@ARTICLE{11456267,
  author={Qian, Quan and Yu, Yang and Qin, Yi and Zhang, Jiusi and Cheng, Yuhua and Chen, Kai},
  journal={IEEE/ASME Transactions on Mechatronics}, 
  title={Asynchronous Joint Distribution Alignment: A New Domain Confusion Mechanism for Fault Transfer Diagnosis}, 
  year={2026},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TMECH.2026.3669233}
  }
```

## ⚠️ Limitations & Future Work

### Limitations

- Hyperparameters (DOA threshold `thre` and HoSDR order `m`) are selected via grid search, lacking adaptive optimization.

- Assumes identical label spaces for source and target domains, cannot be directly applied to partial-set or open-set scenarios.

- Effectiveness in extreme operating conditions (e.g., rapid speed variations) and complex mechanical systems needs further validation.

### Future Work

Future work will focus on adaptive hyperparameter optimization, extending AJDA to address label space inconsistency and validating its robustness under highly complex mechanical systems and operating conditions.

## 📞 Contact

For questions, issues, or cooperation, please contact the paper authors:

- **Corresponding Authors**:  Qian Quan (qian_1998@uestc.edu.cn), Kai Chen (kaichen@uestc.edu.cn)

- **GitHub**: [liguge](https://github.com/liguge)

- **Paper Authors**: Quan Qian, Yang Yu, Yi Qin, Jiusi Zhang, Yuhua Cheng, Kai Chen

## 🙏 Thanks

> *We would like to express our gratitude to all researchers and contributors in the fields of **transfer learning** and **mechanical fault diagnosis**. This project references open-source code from related works, and we appreciate their contributions to the academic community. Special thanks to the authors of MMD, DANN, DTN-JDA, and other transfer learning methods for their open-source implementations, which laid the foundation for this work.*

