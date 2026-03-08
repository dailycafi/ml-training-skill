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

### Training Techniques
- **[Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)** — Compute-optimal training (Hoffmann et al.)
- **[μP (Maximal Update Parameterization)](https://arxiv.org/abs/2203.03466)** — Hyperparameter transfer across model scales
- **[RandAugment](https://arxiv.org/abs/1909.13719)** / **[CutMix](https://arxiv.org/abs/1905.04899)** / **[MixUp](https://arxiv.org/abs/1710.09412)** — Data augmentation strategies
- **[RoPE (Rotary Position Embeddings)](https://arxiv.org/abs/2104.09864)** — Positional encoding for transformers
- **[RMSNorm](https://arxiv.org/abs/1910.07467)** — Efficient layer normalization

### Hardware
- **[NVIDIA DGX Spark (GB10)](https://www.nvidia.com/en-us/products/dgx/spark/)** — Grace Blackwell desktop, bandwidth-optimized training patterns
- **[NVIDIA A100/H100](https://www.nvidia.com/en-us/data-center/)** — Data center GPUs, multi-GPU training

## License

MIT
