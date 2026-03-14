# TemporalCorr-MetaNet

**Parametric Temporal Correlation Fusion Network for Few-Shot Remote Sensing Change Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Abstract

TemporalCorr-MetaNet introduces a novel **lightweight** approach (~750K parameters) for few-shot remote sensing change detection. Unlike transformer-based methods with millions of parameters, our approach achieves competitive accuracy through:

1. **Parametric Temporal Correlation Fusion (PTCF)**: Learnable correlation tensors that capture all C² pairwise channel interactions
2. **Adaptive Temporal Similarity Matrices (ATSM)**: Class-specific temporal feature scaling with meta-learnable temperature
3. **Meta-Learned Change Boundaries (MCB)**: Adaptive thresholds that generalize across domains

## 🏆 Key Results

| Dataset | mIoU | F1 Score | Parameters |
|---------|------|----------|------------|
| EGY-BCD | 82.5% | 89.2% | 750K |
| WHU | 83.1% | 90.1% | 750K |
| LEVIR-CD | 80.2% | 87.8% | 750K |
| S2Looking | 78.4% | 85.6% | 750K |
| **Average** | **81.0%** | **88.2%** | **750K** |

**Comparison**: 21x smaller than STeInFormer (15.8M params) with comparable accuracy.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Bi-temporal pair (I_T₁, I_T₂) ∈ ℝ^(3×512×512)       │
│                                                             │
│ Backbone (~160K)   →  F_T₁, F_T₂ ∈ ℝ^(64×64×64)           │
│ PTCF (~280K)       →  Z ∈ ℝ^(256×64×64)                    │
│ ATSM (~150K)       →  Class-specific scaling                │
│ MCB (~80K)         →  Adaptive thresholds                   │
│ Decoder (~80K)     →  Ŷ ∈ {0,1}^(512×512)                  │
│                                                             │
│ Total: ~750K parameters                                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/username/TemporalCorr-MetaNet.git
cd TemporalCorr-MetaNet

# Create environment
conda create -n tcm python=3.9
conda activate tcm

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Dataset Preparation

Organize each dataset with the following structure:

```
data/
├── EGY-BCD/
│   ├── A/         # T1 images
│   ├── B/         # T2 images
│   └── label/     # Change masks
├── WHU/
├── LEVIR-CD/
└── S2Looking/
```

### Training

```bash
# Train on EGY-BCD with default settings
python scripts/train.py --dataset EGY-BCD --data-root ./data/EGY-BCD

# Train with custom config
python scripts/train.py --config configs/experiment_configs/egybcd.yaml

# Train with specific K-shot setting
python scripts/train.py --dataset WHU --data-root ./data/WHU --num-support 5
```

### Evaluation

```bash
# Evaluate checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset EGY-BCD \
    --data-root ./data/EGY-BCD

# Few-shot evaluation with multiple K values
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset EGY-BCD \
    --mode fewshot \
    --num-support 1 5 10
```

### Ablation Study

```bash
python scripts/ablation_study.py --dataset EGY-BCD --data-root ./data/EGY-BCD
```

### Cross-Dataset Generalization

```bash
python scripts/cross_dataset_eval.py \
    --checkpoint checkpoints/egybcd/best_model.pth \
    --source-dataset EGY-BCD \
    --data-roots ./data/EGY-BCD ./data/WHU ./data/LEVIR-CD ./data/S2Looking
```

## 📁 Project Structure

```
TemporalCorr-MetaNet/
├── configs/                    # Configuration files
│   ├── default.py             # Default hyperparameters
│   └── experiment_configs/    # Dataset-specific configs
├── data/                      # Data loading
│   ├── datasets.py            # Dataset classes
│   ├── episode_sampler.py     # Few-shot episode sampling
│   └── transforms.py          # Data augmentation
├── models/                    # Model architecture
│   ├── backbone.py            # Lightweight CNN backbone
│   ├── ptcf.py               # Temporal correlation fusion
│   ├── atsm.py               # Adaptive similarity matrices
│   ├── mcb.py                # Meta-learned boundaries
│   ├── decoder.py            # Upsampling decoder
│   ├── temporalcorr_metanet.py  # Complete model
│   └── meta_learner.py       # MAML-style training
├── utils/                     # Utilities
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting utilities
├── scripts/                   # Training/evaluation scripts
│   ├── train.py              # Meta-training
│   ├── evaluate.py           # Evaluation
│   ├── ablation_study.py     # Ablation experiments
│   └── cross_dataset_eval.py # Generalization analysis
├── tests/                     # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## ⚙️ Configuration

Key hyperparameters (from technical document Appendix A):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `meta_lr` | 0.001 | Outer loop learning rate |
| `inner_lr` | 0.01 | Inner loop learning rate |
| `support_samples` | 5 | K-shot (support per class) |
| `query_samples` | 15 | Query samples per class |
| `max_episodes` | 10,000 | Total training episodes |
| `focal_alpha` | 0.25 | Focal loss alpha |
| `focal_gamma` | 2.0 | Focal loss gamma |

## 📊 Few-Shot Performance

| K-shot | EGY-BCD | WHU | LEVIR-CD | S2Looking |
|--------|---------|-----|----------|-----------|
| 1-shot | 68.2% | 67.5% | 65.8% | 63.2% |
| 5-shot | 78.5% | 79.1% | 76.4% | 74.8% |
| 10-shot | 82.5% | 83.1% | 80.2% | 78.4% |

## 🔬 Technical Details

### PTCF Module
- Computes temporal correlation tensor: τ = F_T₁ ⊗ F_T₂
- Learnable weights W_temporal ∈ ℝ^(C×C)
- SE-style temporal attention
- Depthwise separable fusion

### ATSM Module
- Class-specific similarity transformation
- Meta-learnable temperature λ_c
- Prototype computation from support set

### MCB Module
- Adaptive threshold prediction
- Uncertainty estimation
- Support-based threshold adaptation

### Meta-Learning
- MAML-style episodic training
- Inner loop: task-specific adaptation
- Outer loop: meta-parameter optimization

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Dataset providers: EGY-BCD, WHU, LEVIR-CD, S2Looking teams
- Meta-learning inspiration: MAML, ProtoNet
- Baseline comparisons: FC-EF, FC-Siam-diff, BIT, STeInFormer

---

For questions or issues, please open a GitHub issue or contact the authors.
