# BdSL Research Benchmarks and Dataset Synthesis

This document provides a comprehensive synthesis of contemporary research, academic benchmarks, and publicly available datasets for Bangla Sign Language (BdSL) recognition.

## 1. Source Breakdown & Taxonomy

| Source | Link/Location | Objective | Volume | Signer Diversity | Capture Conditions | Methodology / Benchmark |
|--------|---------------|-----------|--------|------------------|--------------------|-------------------------|
| **BdSL40** | [GitHub Repo](https://github.com/Patchwork53/BdSL40_Dataset_AI_for_Bangla_2.0_Honorable_Mention) | General word-level sign recognition | 611 videos (40 classes, 8-22 clips/class) | Multi-signer | Variable lighting & backgrounds | Video classification, MediaPipe features, LSTMs |
| **Kaggle Bengali Sign Language** | [Kaggle Dataset](https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset) | Character-level image classification | ~10k+ images | Varied | Mostly static images, uniform/varied | CNNs (ResNet, VGG), high accuracy on static signs |
| **ScienceDirect BdSL Data** | [Article](https://www.sciencedirect.com/science/article/pii/S235234092300447X) | Real-world static & dynamic dataset curation | Extensive | High | Controlled and uncontrolled | Hybrid approaches, establishing baseline metrics |
| **ACM BdSL Research** | [Paper](https://dl.acm.org/doi/10.1145/3723178.3723215) | Exploring novel architectures for regional signs | Various | Multi-signer | Real-world | Transformer-based models and spatial attention |
| **Khulna University ECE** | [Publication](https://ku.ac.bd/discipline/ece/research/publication/626/details) | Localized BdSL gesture analysis | Academic scale | Regional subjects | Lab environment | Computer Vision pipelines and baseline ML |
| **INDORE 45-Character Dataset** | [IEEE DataPort](https://ieee-dataport.org/documents/45-character-bangla-sign-language-indore-dataset) | Comprehensive 45-character alphabet classification | Large image set | Varied | Controlled | Dense CNN classification |
| **BAUST Lipi Benchmark** | [arXiv:2408.10518](https://arxiv.org/abs/2408.10518) | Deep Learning BdSL Recognition | 18,000 images (36 symbols: 30 consonants, 6 vowels) | Multi-signer | 224x224 pixels | Hybrid CNN-LSTM achieved 97.92% accuracy |

## 2. BdSL Gesture Taxonomy

Sign languages are inherently multi-modal. BdSL gestures can be categorized into three fundamental structures:

### One-handed Static Signs
- Primarily used for alphabet characters (vowels and consonants) and simple digits.
- **Example:** The letters "অ" (A) or "ক" (K), and numbers like "১" (1).
- **Optimal Architecture:** Deep CNNs (e.g., MobileNetV2, ResNet) or basic Landmark classification without temporal dependencies.

### Two-handed Static Signs
- Used for compound letters or more complex static concepts.
- **Example:** Specific conjunctive characters in Bengali.
- **Optimal Architecture:** Similar to one-handed, but requires dual-hand landmark tracking (`max_num_hands=2`) and spatial relative features.

### Dynamic Movement Signs
- Encompasses words, conversational phrases, and assistive requests that require temporal trajectory (movement over time).
- **Example:** Words like "ধন্যবাদ" (Thank You) or "কেমন আছেন" (How are you).
- **Optimal Architecture:** Hybrid CNN-LSTM, MediaPipe sequence classification via Bi-LSTM, or Vision Transformers (ViT) to capture the temporal dimension across a fixed window (e.g., 30 frames).

## 3. Benchmark Accuracy Targets

Based on the synthesized literature, modern BdSL recognition systems should target the following accuracy baselines:

- **Static Alphabet Classification:** > 95%
  - *Reference:* BAUST Lipi dataset achieved **97.92%** using a Hybrid CNN-LSTM model on 36 classes (18,000 images).
- **Dynamic Word/Phrase Classification:** > 85%
  - *Reference:* Sequence modeling on varied backgrounds drops accuracy significantly compared to static signs, but robust LSTM/Transformer architectures operating on skeletal landmarks (like MediaPipe) maintain 85-92% real-time accuracy.
- **Latency Requirement:** < 50ms per frame for real-time assistive translation.
