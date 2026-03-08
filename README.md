# ML Training Skill for Claude Code

A Claude Code skill providing battle-tested PyTorch training patterns across all domains — LLMs, computer vision, diffusion models, medical imaging, protein/drug discovery, spatial omics, and genomics.

## Installation

```bash
# Download the .skill file from Releases, then:
claude install ml-training.skill

# Or clone and symlink:
git clone https://github.com/dailycafi/ml-training-skill.git
ln -s $(pwd)/ml-training-skill ~/.claude/skills/ml-training
```

## What's Included

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — training loops, LR scheduling, mixed precision, memory optimization, debugging |
| `references/architecture.md` | Architecture selection guide (Transformers, CNNs, U-Nets, GNNs, SSMs) |
| `references/optimizers.md` | Deep dive into AdamW, Muon, hybrid MuonAdamW, per-parameter-group configs |
| `references/domain-specific.md` | Domain-specific patterns (vision, diffusion, speech, RL, time series) |
| `references/biomedical.md` | Biomedical ML (molecular GNNs, protein LMs, medical imaging, genomics, spatial omics, clinical NLP) |
| `references/scaling-and-selection.md` | Scaling laws, compute budgets, hardware selection (DGX Spark, A100, H100) |
| `references/experiment-loop.md` | Autonomous experiment loop for rapid ML iteration |

## Triggers

The skill activates when you discuss:
- Training, fine-tuning, or debugging neural networks
- Loss spikes, NaN gradients, convergence issues
- Optimizer choices (AdamW, Muon, SGD)
- Hardware optimization (DGX Spark, multi-GPU, torch.compile)
- Domain-specific architectures (ViT, U-Net, ESM-2, GNNs, MONAI)
- Metrics (Dice, FID, perplexity, BPB)

## References & Acknowledgments

This skill draws from the following sources:

### Core Foundations
- **[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)** — Autonomous experiment loop, single-file training patterns, keep/discard discipline, fixed time-budget experimentation, BPE tokenizer training with [rustbpe](https://github.com/karpathy/rustbpe)
- **[Karpathy's nanochat](https://github.com/karpathy/nanochat)** — Modern LLM pretraining recipes, Muon optimizer, hybrid MuonAdamW, NorMuon variance reduction, cautious weight decay

### Optimizers
- **[Muon Optimizer](https://github.com/KellerJordan/Muon)** — Polar Express orthogonalization, Newton-Schulz iterations for 2D weight matrices
- **[PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)** — Decoupled weight decay, fused step patterns for torch.compile

### Architectures & Frameworks
- **[PyTorch](https://pytorch.org/)** — Core training framework, torch.compile, FSDP, AMP
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — Model architectures, training utilities
- **[MONAI](https://monai.io/)** — Medical imaging framework (3D U-Net, Swin-UNETR, transforms)
- **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** — Self-configuring medical image segmentation
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** — ZeRO optimization stages, distributed training

### Biomedical ML
- **[ESM-2](https://github.com/facebookresearch/esm)** — Protein language models (Meta)
- **[AttentiveFP](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)** — Graph neural networks for molecular property prediction
- **[SchNet](https://github.com/atomistic-machine-learning/schnetpack)** — 3D molecular modeling
- **[RDKit](https://www.rdkit.org/)** — Cheminformatics toolkit, molecular featurization
- **[PyTorch Geometric](https://pyg.org/)** — Graph neural network framework
- **[Scanpy / scverse](https://scanpy.readthedocs.io/)** — Single-cell analysis ecosystem
- **[SpatialData](https://spatialdata.scverse.org/)** — Spatial omics data framework

### Scaling Laws & Training Methodology
- **[Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)** — Compute-optimal training (Hoffmann et al., 2022)
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** — Original scaling laws (Kaplan et al., 2020)
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)** — Data repetition limits (Muennighoff et al., 2023)
- **[μP (Maximal Update Parameterization)](https://arxiv.org/abs/2203.03466)** — Hyperparameter transfer across model scales (Yang et al., 2022)
- **[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)** — Systematic HP tuning (Google Research)
- **[A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)** — Karpathy's debugging methodology (2019)
- **[Schedule-Free Adam](https://arxiv.org/abs/2405.15682)** — Eliminates LR scheduling (Defazio & Mishchenko, 2024)

### Architecture Papers
- **[ViT (An Image is Worth 16x16 Words)](https://arxiv.org/abs/2010.11929)** — Vision Transformer (Dosovitskiy et al., 2020)
- **[ConvNeXt](https://arxiv.org/abs/2201.03545)** — Modern CNN (Liu et al., 2022)
- **[ResNet Strikes Back](https://arxiv.org/abs/2110.00476)** — Training recipe > architecture (Wightman et al., 2021)
- **[Mamba](https://arxiv.org/abs/2312.00752)** — Linear-time sequence modeling (Gu & Dao, 2023)
- **[RWKV](https://arxiv.org/abs/2305.13048)** — RNN-Transformer hybrid (Peng et al., 2023)
- **[RoPE (Rotary Position Embeddings)](https://arxiv.org/abs/2104.09864)** — Positional encoding for transformers
- **[RMSNorm](https://arxiv.org/abs/1910.07467)** — Efficient layer normalization
- **[Flash Attention](https://arxiv.org/abs/2205.14135)** — IO-aware exact attention (Dao et al., 2022)

### Data Augmentation
- **[RandAugment](https://arxiv.org/abs/1909.13719)** / **[CutMix](https://arxiv.org/abs/1905.04899)** / **[MixUp](https://arxiv.org/abs/1710.09412)** — Data augmentation strategies

### Training at Scale
- **[LLaMA](https://arxiv.org/abs/2302.13971)** — Inference-optimal pretraining (Touvron et al., 2023)
- **[PaLM](https://arxiv.org/abs/2204.02311)** — Scaling language modeling (Chowdhery et al., 2022)
- **[OPT](https://arxiv.org/abs/2205.01068)** — Open pre-trained transformers (Zhang et al., 2022)
- **[Cramming](https://arxiv.org/abs/2212.14034)** — Training LM on single GPU in one day (Geiping & Goldstein, 2022)
- **[ML Engineering](https://github.com/stas00/ml-engineering)** — Practical guide (Stas Bekman)

### Hardware
- **[NVIDIA DGX Spark (GB10)](https://www.nvidia.com/en-us/products/dgx/spark/)** — Grace Blackwell desktop, bandwidth-optimized training patterns
- **[NVIDIA A100/H100](https://www.nvidia.com/en-us/data-center/)** — Data center GPUs, multi-GPU training

## License

MIT
