---
name: ml-training
description: >
  Use this skill for ANY neural network training task — training, fine-tuning, or debugging deep learning
  models in PyTorch. Covers all domains: LLMs, vision, diffusion, medical imaging, protein/drug discovery,
  spatial omics, genomics. Trigger on: loss, epochs, learning rate, optimizer, batch size, GPU performance,
  convergence, Dice/FID/perplexity metrics, architecture choices (RMSNorm, RoPE, attention), U-Net, ViT,
  ESM-2, MONAI, nnU-Net, EMA, DGX Spark, torch.compile, DeepSpeed, FSDP, or 训练/fine-tune 模型.
  Do NOT trigger for: pandas data analysis, API serving, unit testing, or scRNA-seq DE analysis.
---

# ML Training Best Practices

This skill provides battle-tested patterns for PyTorch model training across domains — language models,
computer vision, diffusion models, speech, and general supervised learning. Drawn from production
codebases (Karpathy's autoresearch/nanochat, torchvision, HuggingFace) and modern training practice.
The patterns are organized by concern — pick what's relevant to your situation.

## Table of Contents

1. [Architecture Selection Guide](#architecture-selection-guide) — **Start here**: pick the right model for your task
2. [Scaling Laws & Compute Budget](#scaling-laws--compute-budget) — How big, how much data, how long
3. [Training Loop Patterns](#training-loop-patterns)
4. [Optimizer Configuration](#optimizer-configuration)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Mixed Precision & Compilation](#mixed-precision--compilation)
7. [Data Loading](#data-loading)
8. [Model Architecture Patterns](#model-architecture-patterns)
9. [Memory & Performance](#memory--performance)
10. [Hyperparameter Search](#hyperparameter-search)
11. [Experiment Management](#experiment-management)
12. [Debugging Checklist](#debugging-checklist)

**Reference files** (deeper dives, read when needed):
- `references/architecture.md` — Transformer/LLM architecture code patterns
- `references/optimizers.md` — Muon, AdamW hybrid, per-group LR
- `references/domain-specific.md` — Vision, diffusion, contrastive, distributed, checkpointing
- `references/scaling-and-selection.md` — Scaling laws, compute budget tables, decision trees, DGX Spark optimization
- `references/biomedical.md` — Drug discovery, protein models, medical imaging, genomics, clinical NLP
- `references/experiment-loop.md` — Autonomous agent experiment loop (autoresearch-style keep/discard/revert workflow)

---

## Architecture Selection Guide

Before writing training code, pick the right architecture. This decision depends on **data type**, **data scale**, and **compute budget**.

### Quick decision by data type

| Data Type | < 10K samples | 10K-100K | 100K-1M | > 1M |
|-----------|--------------|----------|---------|------|
| **Images** | Pretrained CNN + fine-tune | Fine-tune ViT or CNN | ViT or hybrid from scratch viable | ViT dominates |
| **Text (classification)** | Few-shot LLM or fine-tune BERT | Fine-tune encoder model | Train domain-specific | Scale up model |
| **Text (generation)** | Few-shot prompting | Fine-tune GPT/LLaMA (LoRA) | Continue pretraining + fine-tune | Pretrain from scratch |
| **Tabular** | XGBoost/LightGBM | Still XGBoost | Neural viable (FT-Transformer) | Both competitive |
| **Time series** | ARIMA / simple LSTM | LSTM/GRU or N-BEATS | PatchTST / Informer | Mamba / SSM |
| **Audio** | Pretrained Whisper | Fine-tune AST | Train from scratch | Scale up |
| **Molecules** | Pretrained GNN (SchNet) | Fine-tune molecular LM | Train GNN from scratch | Scale to 3D models |
| **Proteins** | ESM-2 embeddings + head | Fine-tune ESM-2 / ProtTrans | Train protein LM | AlphaFold-scale |
| **Medical images** | Pretrained CNN + fine-tune | nnU-Net (auto-config) | Swin-UNETR / MedSAM | Foundation models |
| **Genomics** | DNABERT-2 fine-tune | HyenaDNA / Enformer | Train from scratch | Evo / Caduceus |

**Key principle**: architecture matters less than training recipe at equal compute. A well-tuned ResNet beats a poorly-tuned ViT (ref: "ResNet Strikes Back", Wightman 2021).

For biomedical domains (molecules, proteins, medical imaging, genomics, clinical NLP), see `references/biomedical.md`.

### Sequence model selection

| Scenario | Best Choice | Why |
|----------|------------|-----|
| General NLP, seq < 4K | Transformer | Best quality, proven |
| Long sequences > 8K, inference cost matters | Mamba / SSM hybrid | O(n) vs O(n²), constant memory at inference |
| Streaming / edge deployment | RWKV or Mamba | RNN-like constant-memory inference |
| Strong in-context learning needed | Transformer | Attention explicitly "looks back" at examples |
| Continuous signals (audio, genomics) | Mamba / S4 | Designed for continuous data |

For detailed decision trees and compute budget tables, see `references/scaling-and-selection.md`.

---

## Scaling Laws & Compute Budget

### Chinchilla rule (Hoffmann et al., 2022)

For **compute-optimal** training: train on **~20 tokens per parameter**.

| Model Size | Compute-Optimal Tokens | Inference-Optimal (100x) |
|-----------|----------------------|--------------------------|
| 125M | 2.5B tokens | 12.5B tokens |
| 1B | 20B tokens | 100B tokens |
| 7B | 140B tokens | 700B tokens |
| 70B | 1.4T tokens | 7T tokens |

**Inference-optimal**: If deployment cost matters more than training cost (most production cases), overtrain smaller models on 100-200 tokens/param. This is the LLaMA philosophy.

### FLOPs estimation

For Transformers: **FLOPs ≈ 6 × N × D** (N = params, D = tokens, covers forward + backward).

### Compute budget quick guide

| Budget (A100 GPU-hours) | From-scratch model size | Fine-tune model size |
|------------------------|------------------------|---------------------|
| < 24h, 1 GPU | < 500M params | Up to 7B with QLoRA |
| 1-7 days, 1 GPU | Up to 1B | Up to 13B with QLoRA |
| 1-7 days, 4-8 GPU | Up to 3B | Up to 70B with QLoRA |
| Weeks, 32+ GPU | 7B+ | Full fine-tune large models |

### Data repetition limit

When data is limited, repeating up to **~4 epochs** is tolerable. Beyond that, returns diminish sharply (ref: Muennighoff et al., 2023 "Scaling Data-Constrained Language Models").

---

## Training Loop Patterns

A well-structured training loop separates concerns: data iteration, forward/backward, optimizer step,
scheduling, and logging. Here's the skeleton that scales from single-GPU experiments to production:

```python
import gc, time
import torch

# === Setup ===
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")  # use TF32 on Ampere+

device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# === Gradient Accumulation ===
tokens_per_microbatch = batch_size * seq_len
assert total_batch_size % tokens_per_microbatch == 0
grad_accum_steps = total_batch_size // tokens_per_microbatch

# === Main Loop ===
total_training_time = 0
step = 0

while not done:
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        (loss / grad_accum_steps).backward()
        x, y = next(train_loader)

    # Schedule, step, zero grad
    update_lr(optimizer, progress)
    optimizer.step()
    model.zero_grad(set_to_none=True)  # more memory-efficient than zero_grad()

    # Fast-fail: abort if loss is exploding
    if loss.item() > 100:
        print("FAIL: loss exploded")
        exit(1)

    torch.cuda.synchronize()
    dt = time.time() - t0

    # Skip first N steps (torch.compile warmup) from timing
    if step > warmup_steps:
        total_training_time += dt

    # GC management: Python's GC causes ~500ms stalls
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
```

### Key principles

- **`set_to_none=True`** on zero_grad frees gradient memory instead of zeroing it — saves memory and is slightly faster.
- **GC freeze/disable** after first step avoids periodic ~500ms stalls from Python's garbage collector. Re-collect every few thousand steps to prevent unbounded growth.
- **Fast-fail** on extreme loss catches divergence immediately instead of wasting the full time budget.
- **Exclude compilation steps** from timing to get accurate throughput/MFU numbers. `torch.compile` JITs on first use, which inflates early step times.
- **Time-based budgets** (vs step-based) make experiments comparable across hardware. Track wall-clock training time and stop when budget is exhausted.
- **`cudnn.benchmark = True`** for fixed-size inputs (vision): `torch.backends.cudnn.benchmark = True` auto-tunes convolution algorithms.
- **Tensor Core alignment**: batch size, hidden dims, seq length should be **multiples of 8** (bf16) or **64** (A100) for optimal Tensor Core utilization.
- **Gradient clipping** is near-universal for Transformers: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. **Exception**: when using Muon optimizer, gradient clipping may be unnecessary — Muon's orthogonalization already normalizes update magnitudes. autoresearch omits clipping entirely with Muon.

---

## Optimizer Configuration

Modern LLM training benefits from using different optimizers for different parameter groups:

```python
def setup_optimizer(model):
    # Group parameters by role
    matrix_params = list(model.transformer.h.parameters())     # Muon
    embedding_params = list(model.transformer.wte.parameters()) # AdamW, higher LR
    lm_head_params = list(model.lm_head.parameters())           # AdamW, lower LR
    scalar_params = [model.resid_lambdas, model.x0_lambdas]     # AdamW, LR=0.5 (x0), 0.005 (resid)

    # Scale LR by model dimension: lr ∝ 1/√(d_model/d_base)
    d_base = 768
    lr_scale = (model_dim / d_base) ** -0.5

    param_groups = [
        {"params": lm_head_params,   "lr": 0.004 * lr_scale, "weight_decay": 0.0},
        {"params": embedding_params, "lr": 0.6 * lr_scale,   "weight_decay": 0.0},
        {"params": matrix_params,    "lr": 0.04,             "weight_decay": 0.2},
        # ... scalar groups with their own LR
    ]
    return optimizer_class(param_groups)
```

### Rules of thumb

- **Embeddings** need higher LR than transformer layers (they're sparse updates)
- **Unembedding (lm_head)** needs lower LR for stability
- **Never weight-decay embeddings** — they're lookup tables, not learned transforms
- **LR scaling by dimension**: `lr * (d_model / d_base)^(-0.5)` keeps training dynamics stable across model sizes
- **Weight decay scheduling**: linearly decay WD to 0 over training — `wd * (1 - progress)`

For Muon optimizer details (polar express orthogonalization, NorMuon variance reduction), see `references/optimizers.md`.

---

## Learning Rate Scheduling

Two approaches, both valid:

### Time-based (autoresearch style)
```python
def get_lr_multiplier(progress):  # progress = elapsed_time / time_budget
    if progress < warmup_ratio:
        return progress / warmup_ratio
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown + (1 - cooldown) * final_lr_frac
```

### Step-based with cosine decay
```python
def get_lr(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### WSD (Warmup-Stable-Decay) — gaining traction for LLM pretraining
```python
def get_lr_wsd(step, total_steps, max_lr, min_lr, warmup_steps, decay_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step < total_steps - decay_steps:
        return max_lr  # stable phase
    else:
        progress = (step - (total_steps - decay_steps)) / decay_steps
        return max_lr - progress * (max_lr - min_lr)
```
WSD is easier to resume training from — you can re-enter the stable phase and continue.

### Guidance

- **Warmup**: 1-5% of training is typical. Helps with training stability early on. **Zero warmup** is also valid with Muon — autoresearch uses `WARMUP_RATIO=0.0` successfully because Muon's orthogonalization provides implicit stability.
- **Warmdown**: 30-50% of training spent in LR decay is common in modern recipes. This matters more than warmup for final performance.
- **Final LR**: 0 (full decay) or ~10% of peak LR both work. Zero is simpler.
- **Schedule-Free Adam** (Defazio & Mishchenko, 2024): eliminates the need for any LR schedule — promising, still being validated at scale.
- Apply the multiplier to all param groups via `group["lr"] = group["initial_lr"] * multiplier`.

---

## Mixed Precision & Compilation

```python
# Environment setup (before any torch imports ideally)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+

# Autocast context for forward pass
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Compile the model for maximum throughput
model = torch.compile(model, dynamic=False, fullgraph=True)
```

### bf16 vs fp16

- **bf16** (preferred on Ampere+/H100): same exponent range as fp32, no loss scaling needed
- **fp16**: needs GradScaler for stability. Use only on older GPUs (V100)
- **Cast embeddings to bf16** explicitly if they're not autocast targets: `model.wte.to(dtype=torch.bfloat16)`

### torch.compile tips

- `dynamic=False` — fixed shapes enable maximum optimization
- `fullgraph=True` — forces full graph capture, errors if it can't. Use when your model has no graph breaks. If compilation fails with fullgraph=True, try without it first — autoresearch uses `dynamic=False` without `fullgraph=True` successfully
- First few steps are slow (JIT compilation) — exclude from timing
- If compilation fails, check for: data-dependent control flow, Python-side tensor ops, unsupported ops

---

## Data Loading

Efficient data loading eliminates the CPU bottleneck:

```python
# Pre-allocate pinned CPU + GPU buffers for zero-copy transfers
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")

# Transfer with non_blocking for overlap
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

### Best-fit packing (no padding)

Instead of padding sequences to fixed length (wastes compute), pack documents tightly:

1. Maintain a buffer of tokenized documents
2. For each row in the batch, greedily fit the largest document that fits the remaining space
3. If nothing fits, crop the shortest document to fill exactly
4. Every row starts with BOS token
5. Result: 100% utilization, no wasted tokens

### Infinite iterators

```python
def make_dataloader(split):
    """Yields (x, y, epoch) forever, cycling through data."""
    epoch = 1
    while True:
        for batch in data_source:
            yield process(batch), epoch
        epoch += 1
```

This pattern avoids DataLoader restart overhead and naturally tracks epochs.

---

## Model Architecture Patterns

### Transformer / LLM

| Component | Recommended | Why |
|-----------|------------|-----|
| Normalization | RMSNorm | ~same quality as LayerNorm, fewer ops (no mean subtraction) |
| Position encoding | RoPE | Relative, extrapolates well, standard practice |
| Attention | Flash Attention 3 | Memory-efficient, faster, exact (not approximate) |
| Window pattern | SSSL (3 short + 1 long) | Saves compute on most layers, last layer sees full context |
| Activation | ReluSquared or SwiGLU | ReluSquared: simple, sparse. SwiGLU: better quality, +50% params in MLP |
| Residual | Learnable scaling + x0 skip | `resid_lambda * x + x0_lambda * x0` stabilizes deep networks |
| Logit cap | Soft capping | `softcap * tanh(logits / softcap)` prevents extreme logits |
| Init | Zero-init output projections | `c_proj` and MLP output init to zero → residual stream starts clean |
| KV heads | GQA (grouped query attention) | Saves memory/compute with minimal quality loss |
| Value embedding | ResFormer-style | Alternating layers get input-dependent value residuals |

### Vision (CNN / ViT)

| Component | Recommended | Why |
|-----------|------------|-----|
| Backbone | ConvNeXt v2 or ViT | ConvNeXt: CNN with modern tricks. ViT: scalable, transfer-friendly |
| Normalization | LayerNorm (ViT) / BatchNorm (CNN) | BatchNorm works well for CNNs; ViT uses LayerNorm |
| Data augmentation | RandAugment + MixUp + CutMix | Essential for vision — more impactful than architecture tweaks |
| Regularization | Stochastic depth + label smoothing | Better generalization than dropout for vision |
| LR schedule | Cosine with warmup | Standard for ImageNet-scale training |
| Optimizer | AdamW (ViT) / SGD+momentum (CNN) | ViTs need adaptive methods; CNNs work fine with SGD |
| Resolution | Progressive resizing | Train small → finetune large saves compute |

### Diffusion Models

| Component | Recommended | Why |
|-----------|------------|-----|
| Architecture | U-Net or DiT | U-Net: classic. DiT: transformer-based, scales better |
| Noise schedule | Cosine or flow matching | Cosine is stable; flow matching is simpler, state-of-art |
| Loss | MSE on noise (epsilon) or v-prediction | v-prediction better at low SNR |
| Conditioning | Cross-attention or AdaLN | Cross-attn for text; AdaLN for class/timestep |
| EMA | Keep EMA model for inference | EMA weights produce higher quality samples |
| Sampling | DDIM / DPM-Solver++ | Faster than DDPM with comparable quality |

### General supervised (classification, regression, tabular)

| Component | Recommended | Why |
|-----------|------------|-----|
| Optimizer | AdamW | Safe default for everything |
| LR finder | Use lr_find before training | Avoids wasting runs on wrong LR |
| Early stopping | Patience 5-10 epochs on val loss | Prevents overfitting without manual tuning |
| Metrics | Task-specific (F1, AUC, RMSE) | Track alongside loss; loss alone can be misleading |
| Class imbalance | Weighted loss or oversampling | Both work; weighted loss is simpler |
| Normalization | BatchNorm for MLP/CNN | Stabilizes training for non-transformer architectures |

For detailed Transformer code patterns, see `references/architecture.md`.
For vision, diffusion, and non-LLM patterns, see `references/domain-specific.md`.

### Weight initialization pattern

```python
def init_weights(model):
    n_embd = model.config.n_embd
    s = 3**0.5 * n_embd**-0.5  # scale by 1/√d

    # Embeddings: normal(0, 1)
    nn.init.normal_(model.wte.weight, mean=0.0, std=1.0)

    # Unembedding: near-zero init
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.001)

    # Transformer blocks
    for block in model.transformer.h:
        nn.init.uniform_(block.attn.c_q.weight, -s, s)
        nn.init.uniform_(block.attn.c_k.weight, -s, s)
        nn.init.uniform_(block.attn.c_v.weight, -s, s)
        nn.init.zeros_(block.attn.c_proj.weight)   # zero-init output
        nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        nn.init.zeros_(block.mlp.c_proj.weight)     # zero-init output
```

---

## Memory & Performance

### Meta device initialization

For large models, initialize on `meta` device (no memory), then materialize:

```python
with torch.device("meta"):
    model = GPT(config)          # zero memory
model.to_empty(device="cuda")    # allocate without init
model.init_weights()              # now initialize properly
```

### MFU calculation

Model FLOPs Utilization = actual throughput / theoretical peak:

```python
def compute_mfu(model_flops_per_token, batch_tokens, step_time, gpu_peak_flops):
    achieved_flops = model_flops_per_token * batch_tokens / step_time
    return achieved_flops / gpu_peak_flops

# Reference peaks (bf16):
# H100 SXM: 989.5 TFLOPS
# A100 SXM: 312 TFLOPS
# RTX 4090: 165 TFLOPS
# DGX Spark GB10: see references/scaling-and-selection.md (bandwidth-limited)
```

Good MFU targets: >30% is decent, >40% is good, >50% is excellent for single-GPU training.

### Peak VRAM tracking

```python
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
```

Log this in your experiment results to catch memory regressions.

### Efficient batched parameter updates

When updating groups of same-shape parameters (common in Muon), stack them for vectorized ops:

```python
stacked = torch.stack(param_list)        # batch the params
# ... do vectorized update on stacked ...
torch._foreach_copy_(param_list, list(stacked.unbind(0)))  # scatter back
```

---

## Hyperparameter Search

Based on Google's Deep Learning Tuning Playbook — a systematic approach to HP tuning.

### Priority order (tune these first)

1. **Learning rate** — most important HP. Always tune first.
2. **Batch size** — use the largest that fits in memory. It's a speed knob, not a quality knob.
3. **Weight decay** — 0.01-0.1 for AdamW. Higher (0.1) for larger models.
4. **Warmup steps** — 1-5% of training.
5. **Architecture HPs** (depth, width, heads) — only after training recipe is stable.

### Search strategy

- **Quasi-random search** (not grid search) — covers the space better with fewer trials
- **Start simple, add complexity** — get a baseline working first, then add one thing at a time
- **80% exploration, 20% exploitation** — spend most budget trying diverse configs
- **Never change more than one thing at a time** when diagnosing issues

### µP (Maximal Update Parameterization)

Optimal hyperparameters transfer across model widths with µP (Yang et al., 2022):
1. Tune HPs on a small "base model" (e.g., width=256)
2. Scale up width — same LR and init scale transfer
3. **Caveat**: depth scaling is less clean; µP primarily addresses width

**Pragmatic alternative**: most practitioners use known defaults and scale LR inversely with model size (halve LR when doubling params).

### The 2025 default recipe

For a new Transformer project, start with these and validate before changing:

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (β1=0.9, β2=0.95, eps=1e-10) |
| Weight decay | 0.1 |
| LR schedule | Cosine decay or WSD |
| Peak LR | 3e-4 (scale down for larger models) |
| Warmup | 2-5% of total steps |
| Batch size | As large as GPU allows |
| Precision | bf16 |
| Gradient clipping | max_norm=1.0 |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Position encoding | RoPE |
| Attention | Flash Attention, optionally GQA |
| Dropout | 0 for pretraining, 0.1 for fine-tuning |

---

## Experiment Management

### Structured results logging

Track experiments in a TSV file for easy comparison:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.9979	44.0	keep	baseline
b2c3d4e	0.9932	44.2	keep	increase matrix LR to 0.04
c3d4e5f	1.0050	44.0	discard	switch to GeLU activation
d4e5f6g	0.0000	0.0	crash	double model width (OOM)
```

### The simplicity criterion

When comparing approaches: **all else being equal, simpler is better**. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. This heuristic from autoresearch keeps training code maintainable as experiments accumulate.

### Autonomous experiment loop

For systematic, agent-driven experimentation (fixed time-budget runs, keep/discard/revert workflow), see `references/experiment-loop.md`. This is useful when running many rapid experiments to iterate on architecture or hyperparameters.

### EMA smoothed loss

Raw training loss is noisy. Use exponential moving average for readable logs:

```python
ema_beta = 0.9
smooth_loss = 0
for step in range(num_steps):
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
```

### Evaluation metrics by domain

| Domain | Primary Metric | Secondary | Notes |
|--------|---------------|-----------|-------|
| **LLM** | BPB (bits per byte) | Perplexity | BPB is vocab-size-independent, comparable across tokenizers |
| **Classification** | Accuracy / Top-5 | F1, AUC-ROC | Use macro-F1 for imbalanced classes |
| **Detection** | mAP@0.5:0.95 | mAP@0.5 | COCO-style evaluation |
| **Segmentation** | mIoU | Dice coefficient | Per-class IoU reveals weak spots |
| **Generation** | FID / IS | CLIP score | FID needs >10k samples for stability |
| **Regression** | RMSE / MAE | R² | Log-transform targets if heavily skewed |
| **Retrieval** | Recall@k | MRR, NDCG | k depends on use case |

```python
# General evaluation pattern
@torch.no_grad()
def evaluate(model, val_loader, compute_metrics):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in val_loader:
        preds = model(x)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    model.train()
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
```

#### BPB for language models

```python
@torch.no_grad()
def evaluate_bpb(model, val_loader, token_bytes):
    total_nats, total_bytes = 0.0, 0
    for x, y in val_loader:
        loss_per_token = F.cross_entropy(..., reduction='none').view(-1)
        nbytes = token_bytes[y.view(-1)]
        mask = nbytes > 0
        total_nats += (loss_per_token * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

### Final summary format

Print a structured block for easy parsing:

```
---
val_bpb:          0.997900
training_seconds: 300.1
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
```

---

## Debugging Checklist

When training goes wrong, check these in order:

### Karpathy's recipe (still canonical)

1. **Become one with the data** — visualize, check distributions, find outliers, verify labels
2. **Get end-to-end running first** — verify metrics make sense on a trivial case
3. **Overfit one batch** — if you can't, you have a bug (not a HP problem)
4. **Then regularize** — add regularization only after you can overfit
5. **Tune hyperparameters** — start with known defaults, then search
6. **Squeeze** — ensembles, test-time augmentation, etc.

### Loss exploding / NaN
1. Reduce learning rate (try 3-10x smaller)
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check for inf/nan in inputs: `assert not torch.isnan(x).any()`
4. Add logit soft capping: `softcap * tanh(logits / softcap)`
5. Add QK-norm: normalize queries and keys before attention dot product
6. Verify weight init (zero-init output projections?)
7. Check if loss reduction is correct with gradient accumulation (`loss / grad_accum_steps`)
8. **Loss spike at scale**: skip the bad batch and restart from last checkpoint (PaLM strategy)

### OOM (Out of Memory)
1. Reduce `DEVICE_BATCH_SIZE`, increase `grad_accum_steps` (same effective batch)
2. Enable `PYTORCH_ALLOC_CONF=expandable_segments:True`
3. Use `model.zero_grad(set_to_none=True)`
4. Use meta device init → `to_empty` instead of direct `to(device)`
5. Check for retained computation graphs (detach tensors used only for logging)
6. Use activation checkpointing: `torch.utils.checkpoint.checkpoint()`
7. Try 8-bit optimizer (bitsandbytes): ~30% memory savings on optimizer states
8. **Hidden VRAM consumers**: CUDA kernel preloading (0.5-2GB), `torch.distributed` init (1-2GB)

### Slow training / Low MFU
1. Verify `torch.compile` is active and not falling back
2. Check `torch.set_float32_matmul_precision("high")` is set
3. Use `pin_memory=True` and `non_blocking=True` for data transfers
4. Profile with `torch.profiler` to find bottlenecks
5. Ensure batch size is large enough to saturate GPU
6. GC stalls? Add `gc.freeze(); gc.disable()` after warmup
7. **Tensor Core alignment**: dims should be multiples of 8 (bf16) or 64 (A100)
8. **Data bottleneck test**: double batch size — if throughput doesn't double, data loading is the bottleneck

### Loss plateaus / Slow convergence
1. Learning rate too low — try 2-5x larger
2. Warmup too long — reduce warmup ratio
3. Weight decay too high — try halving it
4. Check that LR schedule is actually being applied (print LR each step)
5. Data issue? Inspect a few batches manually
6. Model too small for the data/task

### Silent failures (hardest to catch)
1. **Data leakage** between train/val — loss looks great, model is useless
2. **Wrong preprocessing at inference** — augmentation/normalization mismatch
3. **Label errors** — even 1-5% noise significantly hurts; use cleanlab to detect
4. **Shuffling bugs** — correlated batches from sequential file loading
5. **Tokenizer mismatch** — using wrong tokenizer with pretrained model

### What to monitor
- **Gradient norms** per step — spike often precedes loss spike
- **Per-layer activation stats** — mean/variance reveals exploding/vanishing activations
- **Dead neurons** — if >50% of ReLU outputs are zero, dying ReLU problem
- **Learning rate** — verify schedule is actually applied (common silent bug)
